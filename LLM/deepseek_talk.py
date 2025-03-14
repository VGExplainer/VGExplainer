import os
import glob
import json
from ollama import chat
from ollama import ChatResponse
import re


if __name__ == '__main__':
    all_files = glob.glob('code_path'+'/*.c')
    for file in all_files:
        with open(file,'r') as f1:
            test_code=f1.read()
        if len(test_code) > 10000:
            continue
        name = file.split('/')[-1].split('.c')[0]+'.json'
        # rag_vgexp
        rag_path = 'VGExplainer_explanation/'+file.split('/')[-1].split('.c')[0]
        if not os.path.exists(rag_path):
            continue
        rag_files = glob.glob(rag_path+'/*.json')
        rag_content = ''
        for rag_file in rag_files:
            with open(rag_file,'r') as rf:
                rag_content += str(json.load(rf))+', '
        
        if os.path.exists('output_path/'+name):
            print('this has been processed!')
            continue

        #explanation
        m1={'role': 'user', 'content': f'''#Role: You are a professional vulnerability analyst. You are good at analyzing vulnerabilities in code line by line and overall, and can provide specific vulnerability information and specific vulnerability line numbers.
        #Task: Your task is to carefully analyze the vulnerabilities in the code and provide specific line numbers for the vulnerabilities. And I will provide you with vulnerability subgraphs that other experts consider for your reference.
        #Requirement: 
        1. The code starts from the introduction of the function on the first line, without a 0th line. Every time a line break(\ n) is encountered, a new line begins. You must ensure that the line count is correct.
        2. You must refer to the vulnerability subgraphs considered by other experts to provide results. These vulnerability lines are combined into a subgraph that contains complete information about the vulnerability.
        3. You should read and analyze the vulnerabilities in the code line by line. Especially, it is important to think about potential vulnerabilities identified by other experts. And review potential vulnerabilities again in conjunction with the overall code to ensure that no vulnerability is missed and that the vulnerabilities found are indeed present.
        4. Organize your thoughts and tell me which line has which type of vulnerability, for example, there is a xxx type vulnerability in line x. Try to tell me all potential vulnerability line numbers, you don't need to tell me how to fix them.
        #The given code: 
        {test_code}
        #Other experts' opinions about vulnerability subgraphs:
        {rag_content}
        '''}
        #extract vul_lines
        response_1: ChatResponse = chat(model='deepseek-r1:32b', messages=[m1])
        print(response_1['message']['content'])
        m2= {'role': 'user', 'content': f'''#Role: You are a professional data processing expert, and you are good at extracting effective and standardized data from text.
        #Task: I am now providing you with the analysis results of a professional vulnerability analyst. Please help me extract the vulnerability line numbers that he believes exist based on his analysis.
        #Requirement: 
        1. Be sure to carefully read the analyst's analysis to ensure that the extracted vulnerability line numbers are accurate and consistent with the analyst's analysis results.
        2. You need to output the extracted vulnerability line numbers in the following format: ```json [num1,num2,...] ```. Ensure that your results are output strictly in accordance with the formatting requirements.
        3. Do not output any information unrelated to the vulnerability line number like how to fix it. Remember again that the output format is ```json [num1,num2,...] ``` , do not make any mistakes!
        #The analysis result is:
        {response_1['message']['content']}
        '''}
        response_2: ChatResponse = chat(model='deepseek-r1:32b', messages=[m2])
        print(response_2['message']['content'])
        pattern = r'```json(\s*\[(?:(-?\d+(?:\.\d+)?(?:[\s,;,]\s*-?\d+(?:\.\d+)?)*)|)\]\s*)``'
        match = re.search(pattern, response_2['message']['content'], flags=re.DOTALL)
        if match:
            result_str = match.group(1).strip()
        else:
            continue
        print(result_str)
        try:
            vul_line_list = json.loads(result_str)
        except:
            continue
        with open('out_path'+name,'w') as f2:
            json.dump(vul_line_list,f2)
            f2.close()