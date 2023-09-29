---
title: "Summary notes from 'ChatGPT Prompt Engineering for Developers' "
published: false
tags: llm python 
sidebar:
  title: "Skip Buttons"
  nav: llm-toc2
description: "Prompts for using LLM for programming part 2 - 5 min read"
---

tldr; 'everything' you can do with LLM to help with your programming part 2, please click on the navigation to the left to hop about.

# Introduction
With the advent of ChatGPT, large language model (LLM) or AI itself is now in heard and to an extent used by many people. There's plenty of use cases, but [deeplearning.ai](https://www.deeplearning.ai/short-courses/) have launched several short courses on generative AI; some of them were on how to use LLM for programming. Two courses that were of interest to me were:

1) [Pair Programming with a LLM](https://www.deeplearning.ai/short-courses/pair-programming-llm/), this was built in collaboration with Google, which utilises PaLM LLM. Summarised in a different post over [here](/2023/09/28/pair-programming-llm/).

2)  [ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) this was built with OpenAI which uses ChatGPT LLM. Summarised in the current post.

Both courses were extremely interesting, and they taught me to write better prompts to make better use of LLM; in this case, to get better help with programming. This post will be compiling all the useful code and prompts as well as their respective use cases from _ChatGPT/OpenAI_'s course. For summary on the Google course, please click [here](/2023/09/28/pair-programming-llm/).

**_Disclaimer_**: LLMs are not perfect, they suffer from 'hallucination' and make things up, so the code they generate should still be scrutinized and tested.

## PaLM LLM API
If interested in actually using PaLM LLM, please follow this [link](https://developers.generativeai.google/tutorials/setup) to get an API for yourself and access it. But I believe these prompts should would with any LLM, though obviously results may differ. 

<a class="anchor" id="import"></a>

# Import and setting up model
The following is to import the model using the API key
```python
import os
from utils import get_api_key
import google.generativeai as palm
from google.api_core import client_options as client_options_lib

palm.configure(
    api_key=get_api_key(),
    transport="rest",
    client_options=client_options_lib.ClientOptions(
        api_endpoint=os.getenv("GOOGLE_API_BASE"),
    )
)

# select model for text generation, which is what we need 
# there's other models available in PaLM
models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
model_bison = models[0]
model_bison

from google.api_core import retry
@retry.Retry() # decorator used to retry connection to google api if it fails
def generate_text(prompt, 
                  model=model_bison, 
                  temperature=0.0):
    "the temperature is 'randomness' of the LLM, higher more random"
    return palm.generate_text(prompt=prompt,
                              model=model,
                              temperature=temperature)
```

<a class="anchor" id="template"></a>

# General Template for Everything
One thing highlight from the speaker is that 'priming' the LLM is really important, example below.

```python
prompt_template = """
{priming}

{question}

{decorator}

Your solution:
"""

priming_text = "You are an expert at writing clear, concise, Python code."
question = "create a doubly linked list"
decorator = "Insert comments for each line of code."

# PUT TOGETHER AND EXECUTE
prompt = prompt_template.format(priming=priming_text,
                                question=question,
                                decorator=decorator)

## Call the API to get the completion (which is the output from the LLM)
completion = generate_text(prompt)
print(completion.result)
```

# The Programming Prompts 
From here on, all 'question' space is the code you wish to get help with. The priming is fixed for the given use case.

The obvious thing left out is 'please write XYZ code for me' because this is about pair-programming.

<a class="anchor" id="improve"></a>

## Improve existing code
```python
prompt_template = """
I don't think this code is the best way to do it in Python, can you help me?

{question}

Please explain, in detail, what you did to improve it.
"""

completion = generate_text(
    prompt = prompt_template.format(question=question)
)
print(completion.result)
```

Alternative decorator:

```python
Please explore multiple ways of solving the problem, and explain each.


Please explore multiple ways of solving the problem, and tell me which is the most Pythonic
```
<a class="anchor" id="simplify"></a>

## Simplify code 

```python
# option 1
prompt_template = """
Can you please simplify this code for a linked list in Python?

{question}

Explain in detail what you did to modify it, and why.
"""

# option 2
prompt_template = """
Can you please simplify this code for a linked list in Python? \n
You are an expert in Pythonic code.

{question}

Please comment each line in detail, \n
and explain in detail what you did to modify it, and why.
"""
```

<a class="anchor" id="test"></a>

## Write test case
This one is interesting as in can create tests for functions you do not originally have, which in turn is basically recommending how to improve your code.

NOTE: It may help to specify that you want the LLM to output "in code" to encourage it to write unit tests instead of just returning test cases in English.

```python
prompt_template = """
Can you please create test cases in code for this Python code?

{question}

Explain in detail what these test cases are designed to achieve.
"""
```

<a class="anchor" id="efficient"></a>

## Make code more efficient
```python
prompt_template = """
Can you please make this code more efficient?

{question}

Explain in detail what you changed and why.
"""
```

<a class="anchor" id="debug"></a>

## Debug your code
```python
prompt_template = """
Can you please help me to debug this code?

{question}

Explain in detail what you found and why it was a bug.
"""
```

<a class="anchor" id="explain"></a>

## Explain complex code base
```python
prompt_template = """
Can you please explain how this code works?

{question}

Use a lot of detail and make it as clear as possible.
"""
```

<a class="anchor" id="document"></a>

## Document a complex code base
```python
prompt_template = """
Please write technical documentation for this code and \n
make it easy for a non swift developer to understand:

{question}

Output the results in markdown
"""
```