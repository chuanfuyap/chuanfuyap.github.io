---
title: "Summary notes from 'ChatGPT Prompt Engineering for Developers' "
published: true
tags: llm python 
sidebar:
  title: "Skip Buttons"
  nav: llm-toc2
description: "Prompts for using LLM for programming part 2 - 5 min read"
---

tldr; principles/strategies for using LLM, please click on the navigation to the left to hop about.

# Introduction
With the advent of ChatGPT, large language model (LLM) or AI itself is now in heard and to an extent used by many people. There's plenty of use cases, but [deeplearning.ai](https://www.deeplearning.ai/short-courses/) have launched several short courses on generative AI; some of them were on how to use LLM. Two courses that were of interest to me were:

1) [Pair Programming with a LLM](https://www.deeplearning.ai/short-courses/pair-programming-llm/), this was built in collaboration with Google, which utilises PaLM LLM. Summarised in a different post over [here](/2023/09/28/pair-programming-llm/).

2)  [ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) this was built with OpenAI which uses ChatGPT LLM. Summarised in the current post.

Both courses were extremely interesting, and they taught me to write better prompts to make better use of LLM; This post will be compiling all the useful prompt strategies from _ChatGPT/OpenAI_'s course. For summary on the Google course, please click [here](/2023/09/28/pair-programming-llm/).

**_Disclaimer_**: LLMs are not perfect, they suffer from 'hallucination' and make things up, so the output should still be scrutinized.

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
- These help let the model know these are separate sections.
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

## Check whether conditions are satisfied
Ask the model to check whether conditions are satisfied in the prompt.

As an example, provide instructions to reformat a given text, ask the model to check whether the text can be reformatted according to instructions.

<a class="anchor" id="tactic4"></a>

## "Few-shot" prompting
Give successful examples of completing tasks, then ask the model to perform the task.

As an example, start a sentence with a few words, and ask the model to complete the sentence. Or start an essay with a writing style, and ask the model to continue writing in that style.

<a class="anchor" id="tactic5"></a>

# Give the model time to “think”

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

<a class="anchor" id="tactic6"></a>

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

Actual solution:
"""
```

<a class="anchor" id="iterative"></a>

# Iterative Prompt Development
Prompting is an iterative process. You start with a simple prompt, and then add more instructions to it to get better results. You rarely get a good prompt from the first try. 

Here are the steps to improve your prompts iteratively:
1. Start with a clear and specific prompt based on an idea.
2. Analyze why the result does not give desire output.
3. Refine the idea and the prompt by focusing on the parts that need improvement. 
4. Repeat.

ALWAYS apply the principles and tactics above

# Things you can do with LLM
There's many applications of LLM, here are some examples and their recommended prompt strategies to maximize output. 

**_NOTE_**: All these tasks below are not limited to just one set of text despite the examples below only use one set of text. Given that we are using the API and coding in python, we can perform the tasks below on a list of text by using a for loop and prompting, as such:

```python
reviews = [review_1, review_2, review_3, review_4]

for i in range(len(reviews)):
    prompt = f"""
    Your task is to generate a short summary of a product \ 
    review from an ecommerce site. 

    Summarize the review below, delimited by triple \
    backticks in at most 20 words. 

    Review: ```{reviews[i]}```
    """

    response = get_completion(prompt)
    print(i, response, "\n")
```

<a class="anchor" id="summarize"></a>

## Prompts for Summarizing Text
We can use LLM to summarize texts, e.g. reviews, scientific articles, and news articles. Here are some recommendations to include in your prompts for summarizing text:
- Set word/sentence/character limit
- Focus on specific topics within the text
- Try "extract" instead of "summarize"

<a class="anchor" id="inferring"></a>

## Inferring from Text
We can also use LLM to infer things from a text (i.e. extract information), such as:
- The sentiment of a text (What is the sentiment of the following product review?)
- The topic of a text (Determine five topics that are being discussed in the following text ...)
- The emotion of a text (Identify a list of emotions...)
- The author of a text
- The style of a text
- The tone of a text
- The genre of a text
- The intent of a text
- The product/company being discussed in a text (What product is being discussed in the following text?)

We can also combined them, e.g.:
```python
prompt = f"""
Identify the following items from the review text: 
- Sentiment (positive or negative)
- Is the reviewer expressing anger? (true or false)
- Item purchased by reviewer
- Company that made the item

The review is delimited with triple backticks. \
Format your response as a JSON object with \
"Sentiment", "Anger", "Item" and "Brand" as the keys.
If the information isn't present, use "unknown" \
as the value.
Make your response as short as possible.
Format the Anger value as a boolean.

Review text: '''{lamp_review}'''
"""
response = get_completion(prompt)
print(response)
```

<a class="anchor" id="transforming"></a>

## Transforming Texts
LLM can be used for text transformation tasks such as:
- language translation
- identify language
- spelling and grammar checking
- tone adjustment (Translate the following from slang to a business letter)
- format conversion. (Translate the following python dictionary from JSON to an HTML \
table with column headers and title: {data_json})

Like abovem we can combined all of the above, for example translating langauge in a given tone or format. 


<a class="anchor" id="expanding"></a>

## Expanding Texts
This is a simple use case where the prompt is a starting point for something, and the LLM will expand and fill in the rest. For example, it could be to write an e-mail to someone on a specific topic such as requesting for help or reminding them something. We can then use all the strategies above to make the e-mail more specific, or to make the e-mail more polite, or to make the e-mail more formal, etc. Most importantly, remember to include all the necessary context in the prompt. 