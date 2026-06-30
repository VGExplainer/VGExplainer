import os
import json
import pexpect
from glob import glob
import time
from tqdm import tqdm

def start_joern():
    """Start Joern with increased memory allocation and return the process object"""
    # Force JVM memory allocation to 8GB or 16GB to prevent OOM on large files like OpenSSL
    os.environ['JAVA_OPTS'] = '-Xmx8G' 
    process = pexpect.spawn('./joern', encoding='utf-8', env=os.environ)
    process.expect('joern>', timeout=120)
    print("[+] Joern started successfully.")
    return process

def run_command(process: pexpect.spawn, cmd: str, timeout=120):
    """Send a command and strictly monitor the process status"""
    process.sendline(cmd)
    
    # Explicitly listen for the normal prompt, EOF (crash), and TIMEOUT
    idx = process.expect(['joern>', pexpect.EOF, pexpect.TIMEOUT], timeout=timeout)
    
    if idx == 0:
        return process.before
    elif idx == 1:
        raise RuntimeError("JOERN_CRASHED") # Trigger restart signal
    elif idx == 2:
        raise RuntimeError("JOERN_TIMEOUT") # Trigger restart signal

def extract_variables(bin_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    joern_process = start_joern()
    bin_files = glob(os.path.join(bin_dir, "*.bin"))
    
    for cpg_bin in bin_files:
        source_name = os.path.basename(cpg_bin).replace('.bin', '')
        
        # Support breakpoint resume: skip if already extracted
        final_dict_path = os.path.join(output_dir, f'{source_name}_dict.json')
        if os.path.exists(final_dict_path):
            continue
            
        print(f"=== Extracting variables for {source_name} ===")
        local_path = os.path.join(output_dir, f'{source_name}_local.json')
        param_path = os.path.join(output_dir, f'{source_name}_param.json')
        
        try:
            # 1. Import CPG
            run_command(joern_process, f'importCpg("{cpg_bin}")')
            
            # 2. Use high-precision extraction logic
            cmd_local = f'cpg.local.filter(_.lineNumber != None).map(c => (c.id, c.lineNumber, c.name, c.code)).toJsonPretty #> "{local_path}"'
            cmd_param = f'cpg.parameter.filter(_.lineNumber != None).map(c => (c.id, c.lineNumber, c.name, c.code)).toJsonPretty #> "{param_path}"'
            
            run_command(joern_process, cmd_local)
            run_command(joern_process, cmd_param)
            
            # 3. Merge JSON files
            time.sleep(0.5) 
            node_dict = {'local': [], 'parameter': []}
            
            if os.path.exists(local_path):
                with open(local_path, 'r') as f:
                    try: node_dict['local'] = json.load(f)
                    except: pass
                os.remove(local_path)
                
            if os.path.exists(param_path):
                with open(param_path, 'r') as f:
                    try: node_dict['parameter'] = json.load(f)
                    except: pass
                os.remove(param_path)

            with open(final_dict_path, 'w') as f:
                json.dump(node_dict, f, indent=4)
                
        except Exception as e:
            err_msg = str(e)
            print(f"[Error] Failed processing {source_name}: {err_msg}")
            
            # Core Fix: If Joern crashes or times out, force kill the old process and start a new one
            if "JOERN_CRASHED" in err_msg or "JOERN_TIMEOUT" in err_msg:
                print("[-] Joern process died or hung. Restarting Joern to process the next file...")
                joern_process.close(force=True)
                joern_process = start_joern()
                
            # Clean up residual temporary files
            if os.path.exists(local_path): os.remove(local_path)
            if os.path.exists(param_path): os.remove(param_path)
            continue

    try:
        joern_process.sendline('exit')
    except:
        pass
    print("[+] Extraction done.")
    

if __name__ == '__main__':
    joern_path = '/home/joern-cli_v4.0.408'
    os.chdir(joern_path)
    BIN_DIR = "/home/Dataset/primevul/2_bin/j4/0_novul_norm"        # Path to .bin folder
    DICT_OUT_DIR = "/home/Dataset/primevul/5_node_dict/0_novul" # Directory for node_dict.json files
    # Run this script within the Joern installation directory
    extract_variables(BIN_DIR, DICT_OUT_DIR)