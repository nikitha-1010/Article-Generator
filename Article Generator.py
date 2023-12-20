#!/usr/bin/env python
# coding: utf-8

# In[5]:


import openai
import langchain
from langchain.prompt import PromptTemplate
from langchain import LLMChain
from langchain.chains import SimpleSequentialChain


# In[2]:


import os
from langchain.llms import OpenAI
llm = OpenAI(openai_api_key="your api key", temperature=0.5)


llm=OpenAI(temperature=0.5)
print(llm.predict("what is the capital of india?"))


# In[ ]:


prompt=PromptTemplate.from_template("what is the capital of {place}")
llm=OpenAI(temperature=0.3)
chain1=LLMChain(llm=llm,prompt=prompt)
prompt=PromptTemplate.from_template("what are the famous places at {capital}")
llm=OpenAI(temperature=0.3)
chain2=LLMChain(llm=llm,prompt=prompt)
chain=SimpleSequentialChain(chains=[chain1, chain2], verbose=True)
chain.run("punjab")


# In[ ]:


#this model is for writing the article based on given title 
llm=OpenAI(temperature=0.7)
template="Generate a blog on {title}"
prompt_Template=PromptTemplate(input_variable=["title"],template=template)
s_chain=LLMChain(llm=llm,prompt=prompt_Template,output_key="article")


# In[3]:


#this model is for writing the summary of the article
llm=OpenAI(temperature=0.7)
template="summarize the article:{article}"
prompt_Template=PromptTemplate(input_variables=["article"],template=template)
r_chain=LLMChain(llm=llm,prompt=prompt_Template,output_key="summary")


# In[4]:


#this is the overall chain which connects the above two chains
overall_chain=SequentialChain(chains=[s_chain,r_chain],input_variables=["title"],output_variables=["article","summary"],verbose=True)
print(overall_chain)


# In[ ]:




