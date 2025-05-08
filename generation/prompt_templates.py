from langchain_core.prompts import PromptTemplate
import yaml
from pathlib import Path
import os


root_path = os.path.dirname(os.path.abspath(__file__))




# Load a .yaml or .yml file
with open(root_path+"/prompt_templates.yaml", "r",encoding="utf-8") as file:
    prompt_config = yaml.safe_load(file)



# Load a .yaml or .yml file
with open(Path(root_path).parent/"config/dev/app_config.yaml", "r",encoding="utf-8") as file:
    classes = yaml.safe_load(file)



functions = ", ".join(classes['functions'])
top_label_intent = ", ".join(classes['top_label_intent'])
intent_mode = ", ".join(classes['intent_mode'])




def get_template(name="main_prompt") -> dict:

    
    if name == "translator":
       return {"translator":PromptTemplate.from_template(prompt_config['translator']['prompt'])}


    elif name == "top_label_intent":
      
        return {"top_label_intent":PromptTemplate.from_template(prompt_config['top_label_intent']['prompt'].replace("{classes}",top_label_intent))}

    elif name=="intent_mode":

           return {"intent_mode":PromptTemplate.from_template(prompt_config['intent_mode']['prompt'].replace("{classes}",intent_mode))}

    elif name=="function_call":

           return {"function_call":PromptTemplate.from_template(prompt_config['function_call']['prompt'].replace("{classes}",functions))}

    elif name=="sentiment_score":

           return {"sentiment_score":PromptTemplate.from_template(prompt_config['sentiment_score']['prompt'])}

    elif name=="follow_up_prompt":

           return {"follow_up_prompt":PromptTemplate.from_template(prompt_config['is_follow_up']['prompt'])}
    
    elif name=="main_agent_prompt":

           return {"main_agent_prompt":PromptTemplate.from_template(prompt_config['main_agent_prompt']['prompt'])}

   
    else:    raise ValueError(f"Unknown template: {name}")

