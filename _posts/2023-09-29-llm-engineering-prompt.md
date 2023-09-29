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

## OpenAI LLM API
If interested in actually using OpenAI LLM, please follow this [link](https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/) for instructions on how to secure an API key. But I believe these prompts should would with any LLM, though obviously results may differ. 

### Further Notes on using the OpenAI API (taken from the notes)

To install the OpenAI Python library:
```
!pip install openai
```

The library needs to be configured with your account's secret key, which is available on the [website](https://platform.openai.com/account/api-keys). 

You can either set it as the `OPENAI_API_KEY` environment variable before using the library:
 ```
 !export OPENAI_API_KEY='sk-...'
 ```

Or, set `openai.api_key` to its value:

```
import openai
openai.api_key = "sk-..."
```

<a class="anchor" id="import"></a>



# Import and setting up model
The following is to import the model using the API key
```python
import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key  = os.getenv('OPENAI_API_KEY')

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]
```


<a class="anchor" id="principle"></a>

# Prompting Principles
- **Principle 1: Write clear and specific instructions**
- **Principle 2: Give the model time to “think”**

## Example Code using OpenAI LLM
```python
text = f"""
You should express what you want a model to do by \ 
providing instructions that are as clear and \ 
specific as you can possibly make them. \ 
This will guide the model towards the desired output, \ 
and reduce the chances of receiving irrelevant \ 
or incorrect responses. Don't confuse writing a \ 
clear prompt with writing a short prompt. \ 
In many cases, longer prompts provide more clarity \ 
and context for the model, which can lead to \ 
more detailed and relevant outputs.
"""
prompt = f"""
Summarize the text delimited by triple backticks \ 
into a single sentence.
```{text}```
"""
response = get_completion(prompt)
print(response)
```

<a class="anchor" id="tactic1"></a>

# Write clear and specific instructions

## Use delimiter
Use delimiters to clearly indicate distinct parts of the input
- Delimiters can be anything like: ```, """, < >, `<tag> </tag>`, `:`

Example
```python
text = f"""
You should express what you want a model to do by \ 
providing instructions that are as clear and \ 
specific as you can possibly make them. \ 
This will guide the model towards the desired output, \ 
and reduce the chances of receiving irrelevant \ 
or incorrect responses. Don't confuse writing a \ 
clear prompt with writing a short prompt. \ 
In many cases, longer prompts provide more clarity \ 
and context for the model, which can lead to \ 
more detailed and relevant outputs.
"""
```

<a class="anchor" id="tactic2"></a>

## Structured output
Ask for a structured output
- This is useful for when you want to get a specific output from the model, such as a list, a table, or a dictionary or in JSON or HTML format.

Example
```python
prompt = f"""
Generate a list of three made-up book titles along \ 
with their authors and genres. 
Provide them in JSON format with the following keys: 
book_id, title, author, genre.
"""
response = get_completion(prompt)
print(response)
```

<a class="anchor" id="tactic3"></a>

## Checl conditions are satisfied
Ask the model to check whether conditions are satisfied in the prompt.

As an example, provide instructions to reformat a given text, ask the model to check whether the text can be reformatted according to instructions.

<a class="anchor" id="tactic4"></a>

## "Few-shot" prompting
Few-shot prompting is a technique that allows you to provide a small amount of training data to the model to help it learn how to perform a task.

As an example, start a sentence with a few words, and ask the model to complete the sentence. Or start an essay with a writing style, and ask the model to continue writing in that style.

<a class="anchor" id="tactic5"></a>

# Give the model time to “think”¶

## Specify the steps
Specify the steps required to complete a task.

Example:
```python
# example 1
prompt_1 = f"""
Perform the following actions: 
1 - Summarize the following text delimited by triple \
backticks with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the following \
keys: french_summary, num_names.

Separate your answers with line breaks.
Text:
```{text}```
"""

# example 2
prompt_2 = f"""
Your task is to perform the following actions: 
1 - Summarize the following text delimited by 
  <> with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the 
  following keys: french_summary, num_names.

Use the following format:
Text: <text to summarize>
Summary: <summary>
Translation: <summary translation>
Names: <list of names in Italian summary>
Output JSON: <json with summary and num_names>

Text: <{text}>
"""
```

## Work out the solution
Instruct the model to work out its own solution before rushing to a conclusion.

```python 

prompt = f"""
Your task is to determine if the student's solution \
is correct or not.
To solve the problem do the following:
- First, work out your own solution to the problem. 
- Then compare your solution to the student's solution \ 
and evaluate if the student's solution is correct or not. 
Don't decide if the student's solution is correct until 
you have done the problem yourself.

Use the following format:
Question:

question here

Student's solution:

student's solution here

Actual solution:

steps to work out the solution and your solution here

Is the student's solution the same as actual solution \
just calculated:

yes or no

Student grade:

correct or incorrect


Question:

I'm building a solar power installation and I need help \
working out the financials. 
- Land costs $100 / square foot
- I can buy solar panels for $250 / square foot
- I negotiated a contract for maintenance that will cost \
me a flat $100k per year, and an additional $10 / square \
foot
What is the total cost for the first year of operations \
as a function of the number of square feet.

Student's solution:

Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 100x
Total cost: 100x + 250x + 100,000 + 100x = 450x + 100,000
```
Actual solution:
"""
```