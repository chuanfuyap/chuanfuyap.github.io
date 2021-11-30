---
title: "Social Distancing Secret Santa: Automating Secret Santa with Python (including sending out e-mails)"
published: false
tags: python automation
---

_tldr; completed script that accepts list of names and sends out e-mails is [here](#complete-code)._

# Backstory
It is that time of the year again, where we partake in the festivities and practice the joy of giving. I have recently joined a new department to continue my career in research. As tradition in the department (or at least I'd like to think this is the case), we have secret santa event. For those that don't know, the idea of the event is that everyone that participates is assigned a random person (that is also participating) and you would than have to buy a present for said person (santee if you will), and the santee would not know who the santa (person giving the gift) is, hence the name, "Secret Santa". Which means, there thre are few key things for this to work, and they are:
- total anonymity
- equal probability of santees being assigned
- ideally a santa would not be assigned to themselves

While I was hoping we'd do a simple draw of papers containing names to assign santa/santees. But alas, I shall blame it on the pandemic, organisers were more interested in setting up computerised version of this, and I guess this is more covid friendly way of doing things. While the organizer sorted out the santa/santee assigning part, I helped with the automated e-mail sending. So I thought why not just compile it all and make a simple `secret_santa.py` python script. BUT! I shall add a little twist with inspiration from [Numberphile](https://www.youtube.com/watch?v=5kC5k5QBqcc).

## The input file
Ideally, you would have a spreadsheet containing names and corresponding e-mail, so you would just assign santees and e-mail the santas that information. It should look something like this:

## Assigning Santa/Santee
There various ways to do this,

But as said, I shall be adding the twist from numberphile, which would be to:
- assign random numbers, but it'll be in pairs (e.g. 1:1, 2:2, 3:3)
- shuffle the numbers (e.g. 2:2, 1:1:, 3:3)
- shift half the pair of number down (e.g. 2:3, 1:2, 3:1)
- that is your santa/santee pair
- then assign the numbers to participant

_I'll be honest, this is just complicating things, but my brain needed a lil bit of exercise_

## Automating E-mail sending
Add bout gmail less secure apps

<a class="anchor" id="complete-code"></a>

## All in one script 
The completed code is below, you just need to modify the e-mail address and password and it'll accept a csv file (excel sheet is possible with `read_excel`, just modify the import line) and pair people up and send out the e-mails. 

```

```


# Appendix 
Before I can demonstrated the code above, I needed a list of random/fake name and e-mails, for this I adapted the code from [here](https://moonbooks.org/Articles/How-to-generate-random-names-first-and-last-names-with-python-/). The key here is installing `names` package, which is easily done with `pip install names`

```
import names
import pandas as pd
import numpy as np

name_email = {}
for i in range(10):
    ## loops through and generate a random male/female name
    rand_name = names.get_full_name(gender=np.random.choice(["male", "female"]))
    ## make fake e-mail
    email = (".").join(rand_name.split()) +"@secretsanta.com"
    name_email[rand_name] = email
    
## export to csv
pd.DataFrame(data={"E-mail":name_email}).to_csv("secret_santa_participants.csv")
```