---
title: "Social Distancing Secret Santa: Automating Secret Santa with Python (including sending out e-mails)"
published: true
tags: python automation
description: "Automating Traditional Festive Social Event - 7 min read"
---
_tldr; completed script that accepts list of names and sends out e-mails is [here](#complete-code)._

# Backstory
It is that time of the year again, where we partake in the festivities and practice the joy of giving. I have recently joined a new department to continue my career in research. As tradition in the department (or at least I'd like to think this is the case), we hold the secret santa event. For those that don't know, the idea of the event is that everyone that participates is assigned a random person (that is also participating) and you would than have to buy a present for said person (santee if you will), and the santee would not know who the santa (person giving the gift) is, hence the name, "Secret Santa". Which means, there are a few key things for this to work, and they are:
- total anonymity
- equal probability of santees being assigned
- ideally a santa would not be assigned to themselves

While I was hoping we'd do a simple draw of papers containing names to assign santa/santees. But alas, I shall blame it on the pandemic, organisers were more interested in setting up computerised version of this, and I guess this is more covid friendly way of doing things. While the organizer sorted out the santa/santee assigning part, I helped with the automated e-mail sending. So I thought why not just compile it all and make a simple `secret_santa.py` python script. BUT! I shall add a little twist with inspiration from [Numberphile](https://www.youtube.com/watch?v=5kC5k5QBqcc).

## Table of Contents

* [Input Data](#input)
    - Explanation of the expected input data.

* [Assigning Santa/Santee](#assign)
    - Code to randomly assign santa/santee.

* [Automate E-mail](#email)
    - Code to automate e-mail sending. 

<a class="anchor" id="input"></a>

## The Input File
Ideally, you would have a spreadsheet containing names and corresponding e-mail, so you would just assign santees and e-mail the santas that information. It should look something like this:

| Names            | E-mail                           |
|:-----------------|:---------------------------------|
| Barbara Hutching | barbara.hutching@secretsanta.com |
| Cedrick Rivard   | cedrick.rivard@secretsanta.com   |
| Debra Blanco     | debra.blanco@secretsanta.com     |

_
If you don't or havn't started collecting participants, I recommend sharing a Google spreadsheet or OneDrive Excel for people to put their names/e-mails in, and you can extract that information out easily. 

And you would import as such

```python
names = pd.read_csv("secret_santa_participants.csv", index_col=0) ## in this case I had the name in first column
```

<a class="anchor" id="assign"></a>

## Assigning Santa/Santee
There various ways to do this, such as a simple shuffle of names and assigning that as a new column, then making quick check with `==` to make sure there's nobody assigned to themselves. 

But as said, I shall be adding the twist from numberphile, which would be to:
- assign random numbers, but it'll be in pairs (e.g. 1:1, 2:2, 3:3)
- shuffle the numbers (e.g. 2:2, 1:1:, 3:3)
- shift half the pair of number down (e.g. 2:3, 1:2, 3:1)
- that is your santa/santee pair
- then assign the numbers to the participants

_I'll be honest, this is just complicating things, but my brain needed a lil bit of exercise, and I really enjoyed that video_

I'll be demonstrating with pandas, only just because it lets me print out nice little markdown tables for visualisation (which it appears to not be working very well with my blogs, and that is a problem for another day/post):

```python
import numpy as np
df=pd.DataFrame(np.arange(names.shape[0])) #I used the total number of names for size of numbers to generate. 
df[1] = df[0]
```

You first have this:

|col1 |col2 |
|----:|----:|
|   0 |   0 |
|   1 |   1 |
|   2 |   2 |

Now shuffle them and move one row down (shift, if you will) for one of the columns.

```python
### shuffle
df = shuffle(df)
### shift row down and move bottom to the top
newfirst = df.loc[df.index[-1], [1]].values
shuffled = np.concatenate((newfirst, df[1].shift().dropna().values))
### reassign column values to make it 'permanent'
df[1] = shuffled
```

And we have this:

|col1 |col2 |
|----:|----:|
|   9 |   1 |
|   0 |   9 |
|   6 |   0 |

_note: the example random numbers had 10, I have been displaying 3 to save space_

Then we shuffle names and assign the numbers and map it back to the dataframe above and subsequently map back to the name list
```python
## shuffle names and give numbers
namenum = {}
for i, n in enumerate(shuffle(names.index)):
    namenum[i] = n
## map names to numbers
df[0] = df[0].map(namenum)
df[1] = df[1].map(namenum)
## map back to namelist
santa = df.set_index(0).to_dict()[1]
names["Santee"] = names.reset_index()["index"].map(santa).values
```

And voila, now you can send out e-mails telling everyone who to buy gifts for.

| Santa        | E-mail                       | Santee          |
|:-------------|:-----------------------------|:----------------|
| Bobbi Torain | bobbi.torain@secretsanta.com | Eric Germain    |
| Eric Germain | eric.germain@secretsanta.com | Marian Farquhar |
| Helen Green  | helen.green@secretsanta.com  | Bobbi Torain    |

<a class="anchor" id="email"></a>

## Automating E-mail sending
Now I'll keep this section a lot shorter, but the gist is, I highly recommend making a throwaway e-mail account with Gmail, as for this to work, you need to enable access to less secure apps for that account by going to this [link](https://www.google.com/settings/security/lesssecureapps). Additionally, you get the added bonus of roleplaying by naming your account Santa Claus. 

This is the code that send e-mail with python code, be sure to replace your username and password, I encode it `utf-8` because that's what the server accepts, it should be fine without the forced encoding. The gist is that it uses `smtpblib` is the built-in python library for interacting with e-mail servers and sending e-mails, while the ssl is used to encyption/security stuff. 

I am sure this code can be cleaned up to perform `server.login()` once and send multiple e-mails, but I "borrowed" the code from [here](https://realpython.com/python-send-email/) and did everything in 10 minutes, so I didn't clean it up. 

```python
import smtplib, ssl

def send_mail(receiver_email, subject, message, 
              username='secretsanta@gmail.com', password='password', port=587,
             smtp_server = "smtp.gmail.com"):
    
    sendmsg = "Subject: {}\n\n{}".format(subject, message)
    ## security purpose
    context = ssl.create_default_context()
    ## access server
    with smtplib.SMTP(smtp_server, port) as server:
        ## start encryption
        server.starttls(context=context)
        ## login
        server.login(username, password)
        ## send e-mail
        server.sendmail(username, receiver_email, sendmsg.encode("utf-8"))
```

Now we put the send_email and the list of names together to send things out. 

```python
message = """\
Dear {}, 

Hohoho, Merry Christmas!

You have been randomly assigned a Secret Santa. Your recipient is:

{}
 
Please wrap their presents, write the recipient's name and sign from Santa. 

Regards,
Santa Claus"""

for row in names.iterrows():
    ix = row[0]
    values = row[1]
    ## send mail
    send_mail(values["E-mail"], "Secret Santa 20XX ", message.format(ix, values["Santee"]))

    ## put a little sleep timer as per usual automation 'politeness'
    time.sleep(1)
```

And then you can just check the sent mailbox to see it all go out.

If you reached the end, thanks for reading, hope you enjoyed it and hope it helped you. 

<a class="anchor" id="complete-code"></a>

## All in one script 
The completed code is below, you just need to modify the e-mail address and password and it'll accept a csv file (excel sheet is possible with `read_excel`, just modify the import line) and pair people up and send out the e-mails. 

NOTE: for the e-mail, you need to allow access to less secure app! do not forget!

Here you go, have fun spreading joy!

```python
### everything
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import smtplib, ssl
import time

## import input
names = pd.read_csv("secret_santa_participants.csv", index_col=0) ## in this case I had the data ordered with first column
## assign
df=pd.DataFrame(np.arange(names.shape[0]))
df[1] = df[0]

df = shuffle(df)
newfirst = df.loc[df.index[-1], [1]].values
shuffled = np.concatenate((newfirst, df[1].shift().dropna().values))
df[1] = shuffled

namenum = {}
for i, n in enumerate(shuffle(names.index)):
    namenum[i] = n
df[0] = df[0].map(namenum)
df[1] = df[1].map(namenum)

santa = df.set_index(0).to_dict()[1]
names["Santee"] = names.reset_index()["index"].map(santa).values

def send_mail(receiver_email, subject, message, 
              username='secretsanta@gmail.com', password='password', port=587,
             smtp_server = "smtp.gmail.com"):
    
    sendmsg = "Subject: {}\n\n{}".format(subject, message)
    
    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, port) as server:
        server.starttls(context=context)
        server.login(username, password)
        server.sendmail(username, receiver_email, sendmsg.encode("utf-8"))
        
        
message = """\
Dear {}, 

Hohoho, Merry Christmas!

You have been randomly assigned a Secret Santa. Your recipient is:

{}
 
Please wrap their presents, write the recipient's name and sign from Santa. 

Regards,
Santa Claus"""

for row in names.iterrows():
    ix = row[0]
    values = row[1]
    ## send mail
    send_mail(values["E-mail"], "Secret Santa 20XX ", message.format(ix, values["Santee"]))

    ## put a little sleep timer as per usual automation 'politeness'
    time.sleep(1)
```
_I know this isn't quite `secret_santa.py` but, you can copy paste and run in any python IDE_

# Appendix 
Before I can demonstrated the code above, I needed a list of random/fake name and e-mails, for this I adapted the code from [here](https://moonbooks.org/Articles/How-to-generate-random-names-first-and-last-names-with-python-/). The key here is installing `names` package, which is easily done with `pip install names`

```python
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