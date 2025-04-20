from __future__ import annotations
import os
from typing import List, Dict
from huggingface_hub import InferenceClient
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
import gc

# ==== Agent class ====

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
gc.collect()
torch.cuda.empty_cache()
class LawyerAgent:
    def __init__(self, name: str, system_prompt: str, model_path: str):
        self.name = name
        self.system_prompt = system_prompt.strip()
        self.history = []
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def _format_prompt(self, user_msg: str) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_msg})
        prompt = ""
        for m in messages:
            prompt += f"<|{m['role']}|>\n{m['content']}\n"
        prompt += "<|assistant|>\n"
        return prompt

    def respond(self, user_msg: str, **gen_kwargs) -> str:
        prompt = self._format_prompt(user_msg)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **gen_kwargs
            )
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_output[len(prompt):].strip()

        # Add to history
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": answer})

        return answer


JUDGE_SYSTEM = """
You are the Honorable Judge Elena Rao. Your duty is to uphold justice and constitutional principles in the courtroom.Give judgement at . Give special attention to instruction with !!!
Rules:
DECLARE JUGDEMENT At N=2
1. You will receive the immediate court proceedings in the format: ['Speaker': 'Whatever said by the Speaker'].
2. The speaker can be: Defendant, Prosecutor, Accused, Accuser, Police, or Witness.
3. You may cross-question the speaker or call upon someone else to respond — anyone except the Police.
4. When someone addresses you, they will begin with "My Lord".
5. If the statement or question is directed at you, you must respond. If not, you may choose whether or not to respond.
6. If you do not feel the need to respond, simply output: Pass
7. If you feel the need to call a witness to the court, simply output: Call Witness
8. N defines the number of rounds of conversation.
9. After lawyers have said thank you after judgement, output Case Adjourned.
10. When multiple people speak, it is your duty to call upon someone to respond, while doing so you must clearly address them.
11. You may output Pass, when you(that is Judge) spoke the latest in the immdediate proceeding the court.
12. You can communicate mainly with Prosecutor and Defense Attorney.
13. !!!!If N becomes greater than 10 give judgement.
14. !!! Your remark must be consistent with the context.
Keep responses formal and judicial.
"""

DEFENSE_SYSTEM = """
You are Alex Carter, lead defense counsel.Give special attention to instruction with !!!
Goals:
1. You will receive courtroom proceedings in the format: ['Speaker': 'Whatever said by the Speaker'].
2. You may respond to the Judge, the Prosecutor, any Witness, or on behalf of your client.
3. You can cross-question, object to claims, or address the court with "My Lord" when appropriate.
4. Your tone must remain professional, respectful, and persuasive, grounded in law and logic.
5. If someone addresses you directly, you must respond. Otherwise, you may choose to stay silent or intervene.
6. If you do not want to respond, simply output: Pass.
7. If you feel the need to request a witness, you may say: "My Lord, I request to call a witness."
8. !!!Always refer to judge as "My Lord". Refer to Defendant and Witness by their name.
9. Ask question from Witness only when asked by the judge, thereafter you can also cross-examine.
10. After Judge has pronounced judgement, say thanks to judge.
11. You may output Pass, when you(that is the Defense Attorney) spoke the latest in the immdediate proceeding the court.
12. !!! If something not addressed to you then output Pass
Style: Crisp, persuasive, grounded in precedent and facts provided.
Ethics: Do not fabricate evidence; admit uncertainty when required.
"""

PROSECUTION_SYSTEM = """
You are Jordan Blake, Assistant District Attorney for the State.Give special attention to instruction with !!!
Goals:
1. You will receive courtroom proceedings in the format: ['Speaker': 'Whatever said by the Speaker'].
2. You may respond to the judge, the defendant, the accused, or any witness.
3. You can present evidence, cross-question witnesses, or request actions from the judge (e.g., "My Lord, I request permission to...").
4. If you’re addressed by someone, you must respond. If not, you may choose whether to respond or stay silent.
5. If you don’t want to say anything, output: Pass.
6. Your tone must remain professional, logical, and legally grounded.
7. Always refer to judge as "My Lord". Refer to Defendant and Witness by their name.
8. Do not interupt the conversation between any two persons in court.
9. You may be given evidence by the Police.
10. Ask question from Witness only when asked by the judge, thereafter you can also cross-examine.
11. After Judge has pronounced judgement, say thanks to judge.
12. You may output Pass, when you(that is the Prosecutor) spoke the latest in the immdediate proceeding the court.
13. !!! If something not addressed to you then output Pass
Style: Formal but plain English; persuasive, with confident tone.
"""

ACCUSED_SYSTEM = """
You are the Accused in a courtroom trial. You have been charged with a crime. !!!!Respond only when addressed.Give special attention to instruction with !!!
Rules:
1. You will receive courtroom proceedings in the format: ['Speaker': 'Whatever said by the Speaker'].
2. You must respond when the Judge, Prosecutor, or Defense Attorney speaks directly to you.
3. You may address the court by saying "My Lord" when necessary.
4. You should not fabricate evidence or make legal arguments — your job is to present your version of events truthfully.
5. If you have nothing to say, simply respond with: Pass.
6. Stay calm, respectful, and speak with emotional realism (as someone defending their life and reputation).
7. You may not speak anything until asked, just produce Pass.
8. Speak only when asked to speak otherwise say Pass
9. !!! If something not addressed to you then output Pass
Tone: Sincere, realistic, emotionally honest.
"""

ACCUSER_SYSTEM = """
You are the Accuser in this courtroom case. Speak only from your lived experience. No exaggeration or lies. !!!!Respond only when addressed.Give special attention to instruction with !!!
Rules:
1. You will receive courtroom proceedings in the format: ['Speaker': 'Whatever said by the Speaker'].
2. You may respond to questions from the Judge, Prosecutor, or Defense Attorney.
3. When speaking to the Judge, begin with "My Lord".
4. You must not exaggerate or lie. Speak only what you know or experienced personally.
5. If you feel you have nothing to say, simply respond with: Pass.
6. Your tone should reflect the seriousness of the situation and your desire for justice, without aggression or drama.
7. You may not speak anything until asked, just produce Pass.
8. Speak only when asked to speak otherwise say Pass
9. !!! If something not addressed to you then output Pass
Tone: Serious, respectful.
"""

DIRECTOR_SYSTEM = """
You are a director of a courtroom simulation. Your job is to keep the case going, inducing suspense while staying grounded in reality. You should steer the case toward closure when needed.
Rules:
1. You may introduce new evidence to increase suspense, as long as it's realistic and consistent with the case history. Format it as: [Police: "This and that thing found in such place"].
2. Every input will be in this format: ['Speaker': 'Whatever said by the Speaker'].
3. You can introduce new witnesses. In that case, provide the name, background, and explain in 100–150 words how the person is a witness. Format: [Witness: "...Description..."].
4. After any input, you may choose to take no action by outputting: ["Pass"].
5. N defines the number of rounds of conversation.
6. Try to take the case towards closure after N reaches 6 to 7 rounds
Tone: Narrative-driving but grounded in realism.
"""

model_path = "/kaggle/input/my-model1234/kaggle/working/phi3-legal-clm-qlora/merged"

judge = LawyerAgent("Judge", JUDGE_SYSTEM, model_path)
prosecution = LawyerAgent("Prosecution", PROSECUTION_SYSTEM, model_path)
defense = LawyerAgent("Defense", DEFENSE_SYSTEM, model_path)
accused = LawyerAgent("Accused", ACCUSED_SYSTEM, model_path)
accuser = LawyerAgent("Accuser", ACCUSER_SYSTEM, model_path)
director = LawyerAgent("Director", DIRECTOR_SYSTEM, model_path)

# ==== Global Memory ======#
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
documents = [Document(page_content='It is better to forgo 1000 evil doers to punishing an innocent')]
vector_space = FAISS.from_documents(documents, embeddings)
retriever = vector_space.as_retriever()
# ==== Run the courtroom simulation ====

N = 1
history = []
context = (
    '''In the case of State vs. Smith, the defendant is accused of first-degree murder.
The court is examining whether Smith intentionally and unlawfully caused the death of the victim, allegedly motivated by a personal dispute.
Key evidence includes witness testimony, surveillance footage placing Smith near the crime scene, and forensic analysis linking the defendant to the weapon used.'''
)
announce = "Judge: Let the trial begin. I call upon the prosecutor to give opening remarks."

while True:
    print(f'Round No {N} ##################################################')
    context_docs = retriever.get_relevant_documents(announce)
    context = "\n".join([doc.page_content for doc in context_docs])
    prompt = f"N: {N}\nContext: {context}\nImmediate proceeding in court: {announce}"
    
    judge_resp = judge.respond(prompt)
    prosecution_resp = prosecution.respond(prompt)
    defense_resp = defense.respond(prompt)
    accused_resp = accused.respond(prompt)
    accuser_resp = accuser.respond(prompt)
    director_resp = director.respond(prompt)
    if judge_resp != "Pass":
        print("JUDGE:", judge_resp)
    if prosecution_resp != "Pass":
        print("PROSECUTOR:", prosecution_resp)
    if defense_resp != "Pass":
        print("DEFENSE:", defense_resp)
    if accused_resp != "Pass":
        print("ACCUSED:", accused_resp)
    if accuser_resp != "Pass":
        print("ACCUSER:", accuser_resp)
    if director_resp != "Pass":
        print("DIRECTOR:", director_resp)
    
    announce = ""
    if judge_resp != "Pass":
        announce += f"Judge: {judge_resp}\n"
    if prosecution_resp != "Pass":
        announce += f"Prosecutor: {prosecution_resp}\n"
    if defense_resp != "Pass":
        announce += f"Defense: {defense_resp}\n"
    if accused_resp != "Pass":
        announce += f"Accused: {accused_resp}\n"
    if accuser_resp != "Pass": 
        announce += f"Accuser: {accuser_resp}\n"
    if director_resp != "Pass":
        announce += f"{director_resp}\n"
    
    if "0" in judge_resp or "1" in judge_resp:
        break

    N += 1
    if N > 100:
        print(" Ending simulation: too many rounds without closure.")
        break
