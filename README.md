Courtroom Simulation
This is an LLM based AI model that simulates the courtroom environment.


Work Timeline:-
1. Firstly I started searching for research papers in this field, in my Literature Review I found AgentCourt paper by Chen et. el., although not much has been taken from the paper in this project, I got an intuition about how such models work.
2. Then I started watching youtube tutorials about topics like NLU,NLP,RAG,LLM's etc (mainly from freecodecamps and campusX).
3. Thirdly I read the documentation from HuggingFace website.
4. Since I have not dealt much with Large Language Models, I had to take help from models like ChatGPT

My Approach:-
1.	The first thing I tried to do was to search for India Specific Judicial dataset. I found several different datasets on Kaggle, I combined them together to form a QnA based dataset having around 15,382 examples.
2.	Then I started looking for light weight models which can be fine tuned with computation power. One such model was Microsoft/phi-3-mini-4k-instruct.
3.	First I fine tuned the model on large corpus of judicial text using causal language modeling.
4.	Then I tried to further fine tune it on the dataset I prepared, but it was too computationally expensive even with techniques like LORA that I had to abandon the plan.
5.	Once, the model was fine tuned , I started preparing the courtroom environment.
6.	I kept five agents in the courtroom, namely Prosecutor, Defense Lawyer, Accused and Accuser. Besides, these five agents I kept a sixth agent to overlook over these agents, Director.
7.	The work of Director is to maintain flow of the courtroom by introducing different witnesses and evidences in the courtroom. When it introduces evidences, It acts like Police and when it produces witnesses, it acts like the witness itself.
8.	Whenever one agent says something, it is relayed to all agents and agents can decide to Pass or to something. Their responses are recorded and hence evolves the courtroom.
9.	I maintained two levels of memory, one agent specific and other global. Global memory contains the Case Statement and Proceeding of the case.
10.	Context to be passed to each agent is taken from global memory, by running a vector similarity search with respect to what immediately said in the courtroom.
Problems in Solution:-
1.	Although provided thorough instructions, agents still behave erratically sometimes.
2.	For the final PS, I had to make changes in my original code, so there may be few changes in the final code I submit.
3.	To make it faster, in the final code I used a pretrained model.
