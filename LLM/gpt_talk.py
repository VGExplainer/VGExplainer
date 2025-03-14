import os
import glob
import json
from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat.contrib.capabilities import transform_messages
from autogen.agentchat.contrib.capabilities import transforms



OpenAI_key='sk-xxxxxxxxxxxxxx'
llm_config_ve = {"model": "gpt-4o-mini", "temperature":0.3,"api_key": OpenAI_key,'timeout': 300}

if __name__ == '__main__':
    all_files = glob.glob('code_path'+'/*.c')
    context_handling = transform_messages.TransformMessages(
    transforms=[
        transforms.MessageHistoryLimiter(max_messages=10),
        transforms.MessageTokenLimiter(max_tokens_per_message=8192, min_tokens=8192,model="gpt-4o-mini"),
    ])
    for file in all_files:
        with open(file,'r') as f1:
            test_code=f1.read()
        if len(test_code) > 8192:
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


        vulnerability_explainer = AssistantAgent(
        name="vulnerability_explainer",
        system_message = '''You are a professional vulnerability analyst, and your task is to analyze all potential vulnerabilities in the given code.
        Note that you only need to provide the number of lines with vulnerabilities. You don't need to output any other information.
        Note that the code starts from the line introduced by the function as the first line, and every time a line break('\ n') is encountered, a new line begins, making sure not to count from the line.
        Note that the number of lines starts from 1 and there is no 0 line. 
        Your answer must be rigorous and accurate, and fabrication is prohibited.
        The results are presented in a list format as follows: [...,...,...]. No other information needs to be output.
        ''',
        llm_config=llm_config_ve,
        description='Who can accurately identify potential vulnerabilities.',
        )
        
        vulnerability_explainer_rag = AssistantAgent(
        name="vulnerability_explainer",
        system_message = '''#Role: You are a professional vulnerability analyst. You are good at analyzing vulnerabilities in code line by line and overall, and can provide specific vulnerability line numbers.
        #Task: Your task is to carefully analyze the vulnerabilities in the code and provide specific line numbers for the vulnerabilities. And I will provide you with vulnerability subgraphs that other experts consider for your reference.
        #Requirement: 
        1. The code starts from the introduction of the function on the first line, without a 0th line. Every time a line break(\ n) is encountered, a new line begins. You must ensure that the line count is correct.
        2. You must refer to the vulnerability subgraphs considered by other experts to provide results. These vulnerability lines are combined into a subgraph that contains complete information about the vulnerability.
        3. You should read and analyze the vulnerabilities in the code line by line. Especially, it is important to think about potential vulnerabilities identified by other experts. And review potential vulnerabilities again in conjunction with the overall code to ensure that no vulnerability is missed and that the vulnerabilities found are indeed present.
        4. Tell user vulnerability line numbers as the answer in a list format as follows: [...,...,...]. No other information needs to be output.
        5. Your answer must be rigorous and accurate, and fabrication is prohibited.
        ''',
        llm_config=llm_config_ve,
        description='Who can accurately identify potential vulnerabilities.',
        )
        
        user = UserProxyAgent(
            name="user",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            human_input_mode="ALWAYS",
            system_message="user. You are a human admin. You ask the question.",
            llm_config=False,
            code_execution_config=False,
        )
        context_handling.add_to_agent(vulnerability_explainer_rag)
        user.send(test_code, vulnerability_explainer_rag, request_reply=False, silent=True)
        res = user.initiate_chat(
        recipient=vulnerability_explainer_rag,
        # message=f'''Please tell me the vulnerability of the code I have given. Ensure that the number of lines is correct, starting from the function line as the first line each time, and encountering a new line break(\ n) is a new line. There is no 0 line, all lines start from 1. Your answer must be rigorous and accurate, and fabrication is prohibited.''',
        message=f'''Please tell me the vulnerability line numbers of the code I have given in a list format as follows: [...,...,...]. No other information needs to be output. Other experts consider potential vulnerability subgraphs: {rag_content}''',
        clear_history=False,
        max_turns=1)
        vul_line = res.chat_history[2]["content"]
        vul_line_list = json.loads(vul_line)
        with open('output_path/'+name,'w') as f2:
            json.dump(vul_line_list,f2)
            f2.close()
        del vulnerability_explainer_rag
        del user