import os, sys
import glob
import argparse
from multiprocessing import Pool
from functools import partial
import subprocess
import time


def get_all_file(path):
    path = path[0]
    file_list = []
    path_list = os.listdir(path)
    for path_tmp in path_list:
        full = path + path_tmp + '/'
        for file in os.listdir(full):
            file_list.append(file)
    return file_list

def parse_options():
    parser = argparse.ArgumentParser(description='Extracting CPGs.')
    parser.add_argument('-i ', '--input', help='The directory path of input', type=str, default='/2_bin/1_vul_norm/')
    parser.add_argument('-o ', '--output', help='The directory path of output', type=str, default='3_export/pdg_norm/1_vul_/')
    parser.add_argument('-t ', '--type', help='The type of procedures: parse or export', type=str, default='export')
    parser.add_argument('-r ', '--repr', help='The type of representation: pdg, ast, cfg, ddg, cdg, cpg, or lineinfo_json', type=str, default='pdg')
    args = parser.parse_args()
    return args

def joern_parse(file, outdir):
    record_txt = os.path.join(outdir, "parse_res.txt")
    if not os.path.exists(record_txt):
        os.system("touch " + record_txt)
    with open(record_txt, 'r') as f:
        rec_list = f.readlines()
    
    name = file.split('/')[-1].split('.cpp')[0]
    if name + '\n' in rec_list:
        print(" ====> already processed: ", name)
        return
    
    print(' ----> now processing: ', name)
    out = outdir + name + '.bin'
    if os.path.exists(out):
        return
    
    os.environ['file'] = str(file)
    # The parsed bin filename will match the source filename
    os.environ['out'] = str(out) 
    os.system('sh joern-parse $file --language c --output $out')
    
    with open(record_txt, 'a+') as f:
        f.writelines(name + '\n')

def joern_export(bin, outdir, repr):
    record_txt = os.path.join(outdir, "export_res.txt")
    if not os.path.exists(record_txt):
        os.system("touch " + record_txt)
    with open(record_txt, 'r') as f:
        rec_list = f.readlines()

    name = bin.split('/')[-1].replace(".bin", "")
    out = os.path.join(outdir, name)
    
    if name + '\n' in rec_list:
        print(" ====> already processed: ", name)
        return
    
    print(' ----> now processing: ', name)
    os.environ['bin'] = str(bin)
    os.environ['out'] = str(out)
    
    # Handle different representation types (PDG, AST, CFG, etc.)
    if repr == 'pdg':
        os.system('sh joern-export $bin' + " --repr " + "pdg" + ' --out $out')
        try:
            pdg_list = os.listdir(out)
            for pdg in pdg_list:
                if pdg.startswith("0-pdg"):
                    file_path = os.path.join(out, pdg)
                    os.system("mv " + file_path + ' ' + out + '.dot')
                    os.system("rm -rf " + out)
                    break
        except:
            pass
    elif repr == 'ast':
        os.system('sh joern-export $bin' + " --repr " + "ast" + ' --out $out')
        try:
            pdg_list = os.listdir(out)
            for pdg in pdg_list:
                if pdg.startswith("0-ast"):
                    file_path = os.path.join(out, pdg)
                    os.system("mv " + file_path + ' ' + out + '.dot')
                    os.system("rm -rf " + out)
                    break
        except:
            pass
    elif repr == 'cfg':
        os.system('sh joern-export $bin' + " --repr " + "cfg" + ' --out $out')
        try:
            pdg_list = os.listdir(out)
            for pdg in pdg_list:
                if pdg.startswith("0-cfg"):
                    file_path = os.path.join(out, pdg)
                    os.system("mv " + file_path + ' ' + out + '.dot')
                    os.system("rm -rf " + out)
                    break
        except:
            pass
    elif repr == 'ddg':
        os.system('sh joern-export $bin' + " --repr " + "ddg" + ' --out $out')
        try:
            pdg_list = os.listdir(out)
            for pdg in pdg_list:
                if pdg.startswith("0-ddg"):
                    file_path = os.path.join(out, pdg)
                    os.system("mv " + file_path + ' ' + out + '.dot')
                    os.system("rm -rf " + out)
                    break
        except:
            pass
    elif repr == 'cdg':
        os.system('sh joern-export $bin' + " --repr " + "cdg" + ' --out $out')
        try:
            pdg_list = os.listdir(out)
            for pdg in pdg_list:
                if pdg.startswith("0-cdg"):
                    file_path = os.path.join(out, pdg)
                    os.system("mv " + file_path + ' ' + out + '.dot')
                    os.system("rm -rf " + out)
                    break
        except:
            pass
    elif repr == 'cpg':
        os.system('sh joern-export $bin' + " --repr " + "cpg14" + ' --out $out')
        try:
            pdg_list = os.listdir(out)
            for pdg in pdg_list:
                if pdg.startswith("0-cpg"):
                    file_path = os.path.join(out, pdg)
                    os.system("mv " + file_path + ' ' + out + '.dot')
                    os.system("rm -rf " + out)
                    break
        except:
            pass
    else:
        # Default behavior: run custom joern script to export JSON
        pwd = os.getcwd()
        if out[-4:] != 'json':
            out += '.json'
        joern_process = subprocess.Popen(["./joern"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, encoding='utf-8')
        import_cpg_cmd = f"importCpg(\"{bin}\")\r"
        script_path = f"{pwd}/graph-for-funcs.sc"
        # Export as JSON
        run_script_cmd = f"cpg.runScript(\"{script_path}\").toString() |> \"{out}\"\r" 
        cmd = import_cpg_cmd + run_script_cmd
        ret , err = joern_process.communicate(cmd)
        print(ret, err)

    len_outdir = len(glob.glob(outdir + '*'))
    print('--------------> len of outdir ', len_outdir)
    with open(record_txt, 'a+') as f:
        f.writelines(name + '\n')

def main():
    joern_path = '/home/joern-cli_v4.0.408'
    os.chdir(joern_path)
    args = parse_options()
    type = args.type
    repr = args.repr

    input_path = args.input
    output_path = args.output

    if input_path[-1] != '/':
        input_path += '/'

    if output_path[-1] != '/':
        output_path += '/'

    # Using 3 processes for parallel execution
    pool_num = 3
    pool = Pool(pool_num)

    if type == 'parse':
        files = glob.glob(input_path + '*.cpp')
        pool.map(partial(joern_parse, outdir = output_path), files)

    elif type == 'export':
        bins = glob.glob(input_path + '*.bin')
        if repr == 'pdg':
            pool.map(partial(joern_export, outdir=output_path, repr=repr), bins)
        else:
            pool.map(partial(joern_export, outdir=output_path, repr=repr), bins)

    else:
        print('Type error!')    

if __name__ == '__main__':
    main()