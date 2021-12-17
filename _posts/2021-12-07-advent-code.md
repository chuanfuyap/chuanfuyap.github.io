---
title: "Advent of Code 2021: Python"
published: true
tags: python advent-code-2021
description: "solutions to advent of code 2021 in python"
---

_tldr; all the solutions are in this [repo](https://github.com/chuanfuyap/adventcode/tree/main/2021) in jupyter-notebook format marked by their days_

_updated 17-12-2021 for day 10's solution_

# It's puzzle time
While I have been coding for many years, I have not joined the advent of code before. The reason for this was that I know it would be a very time consuming endeavour (and it is). I have done some odd coding challenge on [hackerrank](https://www.hackerrank.com) (though mind you, it was not geared towards employment like it is now when I did them), which was how I know how coding challenges can really take up your schedule while being fun, like any hobby would. 

However this year, in a new environment, my new peers suggested we try a few together. So we did, and since I have already started, I feel the need to finish it, I'll be slow at it given how busy I am trying to finish off work before... well Christmas. But I'll most definitely finish it and post all my solutions. 

So a little background of me so you can understand why I approach the puzzles the way I do. While I have been programming in Python before the data science boom, it was really after joining the data science crowd via a bootcamp that I went full pythonista. And my current work can also be branded as "health data science" so my toolkit is all data science related, which includes [pandas](https://pandas.pydata.org) and [numpy](https://numpy.org) and many more but I highlight these two cause I have used them a lot in my solutions. With my origin story out of the way, let the games begin! _Oh by the way, I don't enjoy playing code golf (writing as little lines as possible), so if that's you're here for, I am sorry._

PS: IF you came before Christmas day and see lots of unfinished puzzles, I'll be updating this blog as I solve the them, so please come back if you would like to see my solutions. 

# Table of Contents
* [Day 1](#day1)
* [Day 2](#day2)
* [Day 3](#day3)
* [Day 4](#day4)
* [Day 5](#day5)
* [Day 6](#day6)
* [Day 7](#day7)
* [Day 8](#day8)
* [Day 9](#day9)
* [Day 10](#day10)
* [Day 11](#day11)
* [Day 12](#day12)
* [Day 13](#day13)
* [Day 14](#day14)
* [Day 15](#day15)
* [Day 16](#day16)
* [Day 17](#day17)
* [Day 18](#day18)
* [Day 19](#day19)
* [Day 20](#day20)
* [Day 21](#day21)
* [Day 22](#day22)
* [Day 23](#day23)
* [Day 24](#day24)
* [Day 25](#day25)


<a class="anchor" id="day1"></a>

### Day 1
For day 1, this was pretty straightforward for me, this was essentially time series data manipulation for part 1 and part 2 which I have learned before. 

```python
import pandas as pd 
df = pd.read_csv("day1.txt", names=["value"])
### part 1
(df.value.diff()>0).sum()
### part 2
(df.value.rolling(window=3).sum().diff()>0).sum()
```

<a class="anchor" id="day2"></a>

### Day 2
Day 2 is a simple array manipulation, I have once again opted to solve it with pandas instead of numpy, which came in somewhat handy for part 2. 

```python
import pandas as pd
df = pd.read_csv("day2.txt", sep="\s+", names=["dir","value"])
## part 1
### just sum up respective directions
def direction(whereto): 
    return df[df.dir==whereto].value.sum(0)

direction("forward") * (direction("down")- direction("up"))
```

For part 2, I could have used if else to solve it in a loop, however my peer was really wanting to vectorise the solution, so after some brainstorming here is the solution which pandas came in really handy with new column generation. 

```python
## part 2
## first make new columns for aim which is up/down in their respective negative/positive as a submarine
df["aim"] = df[df.dir=="up"].value*-1
df["aim"] = df["aim"].combine_first(df[df.dir=="down"].value)
## cumulative sum, to get the vertical axis, then shifted to get it next to forward direction
df["aim_shift"] = df.aim.cumsum().shift()
## forward fill in case there's more forward direction after a first forward
df["aim_shift"] = df["aim_shift"] .fillna(method="ffill")
### multiple value with aim to get depth
df["depth"] = df.value*df.aim_shift
### take only forward depth and summed it as above
direction("forward") * df[df.dir=="forward"].depth.sum(0)
```

<a class="anchor" id="day3"></a>

### Day 3
I think day 3 was when things start getting hard for me! Or at least the part 2, and as mentioned at the start, this thing takes time, so while there is definitely better a cleaner way for it, I have decided to leave my hacky solution int with the many if else, I'd grade myself poorly on this one, but hey its a festive game, so I'll take it easy.

```python
### part 1 
df = pd.read_csv("day3.txt", header=None, dtype=str)

df = df.apply(lambda x: list(x[0]), axis=1, result_type="expand")
binary = df.astype(int).sum(0)
gamma = (binary>=500).astype(int).astype(str)
epsilon = (binary<500).astype(int).astype(str)

gamma = int(("").join(gamma.values), 2)
epsilon = int(("").join(epsilon.values), 2)

gamma*epsilon

## part 2
inputdf = pd.read_csv("day3.txt", header=None, dtype=str)
inputdf = inputdf.apply(lambda x: list(x[0]), axis=1, result_type="expand")

def gas(dataframe, o2=True):
    df = dataframe.copy().astype(str)
    
    col = 0
    while df.shape[0] >1:
        a,b = df.groupby(col).size()

        if a==b:
            if o2:
                pick="1"
            else: 
                pick="0"
            df = df[df[col]==pick]
        else:
            if o2:
                common = np.argmax([a,b])
            else:
                common = np.argmin([a,b])
                
            df = df[df[col]==str(common)]
        
        col+=1
    return df

o2 = int(("").join((gas(df).values[0])), 2)
co2 = int(("").join((gas(df, False).values[0])), 2)
o2 * co2
```

<a class="anchor" id="day4"></a>

### Day 4
This one was pretty fun, as I never really had to deal with dataformat like this before, reminding of a little tic-tac-toe machine, but 5x5 instead of 3x3. Since it was a non-standard input, I had to make my own input reader

```python
import numpy as np
import pandas as pd

### import boards
bingoboards = []
with open("day4.txt", "r") as f:
    draws = f.readline().rstrip().split(",")
    
    lines = f.readlines()
    
    for line in lines:
        if line =="\n":
            board = []
        else:
            board.append(line.rstrip().split())
            
        if len(board)==5:
            bingoboards.append(np.array(board))

bingoboards = np.array(bingoboards)
```

And since we need to determine winners within, that means I needed a function for that as well, and this would be used for both parts, since the second one is to find the loser

```python
def chickendinner(boards):
    horizontal = (boards=="GO").sum(1) 
    vertical = (boards=="GO").sum(2) 
    
    checkrow = np.where(horizontal == 5 )    
    checkcol = np.where(vertical == 5 )
    
    if checkrow[0].shape[0] > 0:
        winner = checkrow[0]
    elif checkcol[0].shape[0] > 0:
        winner = checkcol[0]
    else:
        winner = []
    
    return np.array(winner)
```

And the solutions. 

```python
### part 1
win =  np.array([])
ix = 0
p1board = bingoboards.copy()
while not win.size>0:
    p1board[p1board == draws[ix]] = 'GO'
    win = chickendinner(p1board)
    ix+=1
    
pd.DataFrame(p1board[win][0]).replace("GO", 0).astype(int).sum().sum() * int(draws[ix-1])

## part 2
winner=[]
p2board = bingoboards.copy()


for d in draws:
    blist = list(np.arange(0, p2board.shape[0]))
    p2board[p2board == d] = 'GO'
    win = chickendinner(p2board)
    
    if p2board.shape[0]==1:
        win = chickendinner(p2board)
        if win.size>0:
            print(pd.DataFrame(p2board[win][0]).replace("GO", 0).astype(int).sum().sum() * int(d))
            break
    elif win.size>0:
        losers = list(set(blist).difference(set(win)))
        p2board = p2board[losers]
        blist = list(np.arange(0, p2board.shape[0]))
```

<a class="anchor" id="day5"></a>

### Day 5
Day 5 was an interesting one, it reminds me of minesweeper with the numbers on the board. 

```python
import numpy as np
import pandas as pd

zeros = np.zeros((1000,1000))
with open("day5.txt", "r") as f:
    for line in f.readlines():
        a, b = line.rstrip().split("->")
        x1,y1 = [int(i) for i in a.split(",")]
        x2,y2 = [int(i) for i in b.split(",")]
        if x1==x2:
            if y1<y2:
                zeros[x1, y1:y2+1]+=1
            else:
                zeros[x1, y2:y1+1]+=1
        elif y1==y2:
            if x1<x2:
                zeros[x1:x2+1, y1]+=1
            else:
                zeros[x2:x1+1, y1]+=1

answer = (zeros>=2).sum().sum()
print(answer)
```

The challenge here amps up with the need for diagonal lines in part 2. To solve that I had to draw out the diagonal lines to visually see the indices, which occured to me to be just increasing number in 2d manner, that is `([2,3,4,5],[2,3,4,5])` are the indices for a diagonal line going down starting from row/col 2 going down to row/col 5. 

```python
### part 2

zeros = np.zeros((1000,1000))
with open("day5.txt", "r") as f:
    for line in f.readlines():
        a, b = line.rstrip().split("->")
        x1,y1 = [int(i) for i in a.split(",")]
        x2,y2 = [int(i) for i in b.split(",")]
        if x1==x2:
            if y1<y2:
                zeros[x1, y1:y2+1]+=1
            else:
                zeros[x1, y2:y1+1]+=1
        elif y1==y2:
            if x1<x2:
                zeros[x1:x2+1, y1]+=1
            else:
                zeros[x2:x1+1, y1]+=1
        else:
            if x1<x2:
                d1 = np.arange(x1,x2+1)
            elif x1>x2:
                d1 = np.flip(np.arange(x2,x1+1))
            if y1<y2:
                d2 = np.arange(y1,y2+1)
            elif y1>y2:
                d2 = np.flip(np.arange(y2,y1+1))
            diag = (d1,d2) 
            
            zeros[diag]+=1

answer = (zeros>=2).sum().sum()
print(answer)
```

<a class="anchor" id="day6"></a>

### Day 6
Ah day 6, this was a good one, initial reading made me think, oh exponetial growth? I do computational biology, I know exponential growth and boy was I wrong. The solution of part can be just simulated which I did. 

```python
import numpy as np
import pandas as pd
from collections import Counter

## part1
fishies =np.loadtxt("day6.txt", delimiter="," )
for i in range(80):
    ## move counter down by 1
    fishies-=1
    ## give birth
    newbabies = (fishies==-1).sum()
    if newbabies:
        fishies[fishies==-1] =6
        new = np.zeros(newbabies)+8
        fishies = np.concatenate([fishies,new])
print(fishies.shape[0])
```

Then came part 2, I foolishly just went oh I just need to run it for longer? Okay sure, but oof, it was not smart of me. After failing that, I went on the subreddit for the first time, since it was late and I was too tired to use my brain further to see what other's thought, and the reason I am sharing this is because I found some great memes regarding this puzzle like this [one](https://i.redd.it/9gaqnlcl6y381.png) and [this](https://i.redd.it/iykkkbf7py381.jpg). So the next morning, I just calmly thought of the solution which was to count the number of fishes with a counter instead of having them as an element in an array. 

```python
### part 2
fishies =np.loadtxt("day6.txt", delimiter="," ).astype(int)

fishies = dict(Counter(fishies))
for i in range(-1,9):
    if i not in fishies:
        fishies[i]=0 
for i in range(256):
    ## move counter down by 1
    fishies = {i-1 : fishies[i] for i in range(-1,9)}
    ## give birth
    if fishies[-1]>0:
        fishies[8]=fishies[-1]
        fishies[6]+=fishies[-1]
        fishies[-1]=0
    else:
        fishies[8]=0
print(sum(list(fishies.values())))
```

<a class="anchor" id="day7"></a>

### Day 7
I'll be honest, day 7's solution was ruined for me, I left reddit open from the night before, so I basically woke up to a meme with the solution for it, oh well, here's my code for it. 

```python
import numpy as np

### part 1 
crabs = np.loadtxt("day7.txt", delimiter=",")
trackfuel = []

for i in range(crabs.shape[0]):
    fuelcount = np.abs(crabs-i).sum()
    trackfuel.append(fuelcount)
np.min(trackfuel)

crabs = np.loadtxt("day7.txt", delimiter=",")
trackfuel = []
def gauss(n):
    top = n*(n+1)
    btm = 2
    return top/btm
for i in range(crabs.shape[0]):
    fuelcount = np.abs(crabs-i)
    fuelcount = gauss(fuelcount).sum()
    trackfuel.append(fuelcount)
    
np.min(trackfuel)
```

<a class="anchor" id="day8"></a>

### Day 8
So day 8, my goodness, this was tough, probably the most code I have ever written for a puzzle. But first the easy bit, part 1. 

```python
import numpy as np
import pandas as pd 

### part 1 
## reading in data
with open("day8.txt", "r") as f:
    datainput = []
    for x in f.readlines():
        tok = x.split("|")
        datainput.append((tok[0].split(), tok[1].split()))
count=0
for codelist, output in datainput:
    output = [len(o) for o in output]
    for o in output:
        if o in [2,4,3,7]:
            count+=1
print(count)
```

I worked on part 2 for a really long time until I gave up and went to bed, then next morning I sat on the couch just thinking bout it. This was basically running at the back of my mind even while at work. However during lunch time, solution came to me. It was obvious, I have known this can be viewed as cryptography which means code breaking, and the easiest way for this was the frequency of the letters, which reminded me of the scene in the movie Zodiac. This frequency was what I used for deciphering the codes. 

So here is my loooong code to solve this. 

```python
def builddf(numlist):
    nlist = [sorted(x) for x in numlist]
    df = pd.DataFrame(columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', "num", "key"], index=list(range(10)))
    
    for i, o in enumerate(nlist):
        df.loc[i, o] = 1
        key = ("").join(o)
        df.loc[i, "key"] = key
        
    return df

class locksmith():
    def __init__(self):
        self.keys = {} ## decryption key
        self.letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        self.codefreq = None
        self.flipcode = None
        
        numbers = {'abcefg': 0, 'cf': 1,'acdeg': 2,'acdfg': 3,
                     'bcdf': 4,'abdfg': 5,'abdefg': 6,'acf': 7,
                     'abcdefg': 8,'abcdfg': 9}
        
        nlist = list(numbers.keys())

        self.ogdf = builddf(nlist)
        
        for k,v in numbers.items():
            ix = self.ogdf[self.ogdf.key==k].index[0]
            self.ogdf.loc[ix, "num"] = v
        
        self.freqtable = self.ogdf[self.letters].sum().to_dict()
        self.flipfreq = {v:k for k,v in self.freqtable.items()}
        
        self.o7_1 = self.advanceddecrypt7p1(self.ogdf, self.freqtable)
        self.o7_1 = list(self.o7_1)[0]
        self.o7_2 = self.advanceddecrypt7p2(self.freqtable, self.o7_1)
        
    def basicdecrypt(self, code):
        basicnums = [4,6,9]
        self.codefreq = code[self.letters].sum().to_dict()
        self.flipcode = {v:k for k,v in self.codefreq.items()}
        
        for num in basicnums:
            og = self.flipfreq[num]
            cipher = self.flipcode[num]
            self.keys[og]= cipher
            
    def pairfreqix(self, n, df, freq):
        pair = []
        for k,v in freq.items():
            if v == n:
                pair.append(k)
        tmp = df[pair].dropna(axis=0, how="all")
        ix1 = tmp[pair[0]].index[tmp[pair[0]].isnull()][0]
        ix2 = tmp[pair[1]].index[tmp[pair[1]].isnull()][0]
        
        return [ix1,ix2]
    
    def advanceddecrypt7p1(self, code, freq):
        ix = self.pairfreqix(7, code, freq)
        ab = list(code.iloc[ix].key.values)
        ab.sort(key=len)
        a,b = ab
        
        return set(a).difference(b)
    
    def advanceddecrypt7p2(self, freq, p1):

        for k,v in freq.items():
            if v == 7 and k != p1:
                return k
    
    def advanceddecrypt7(self, code):
        c7_1 = self.advanceddecrypt7p1(code, self.codefreq)
        c7_1 = list(c7_1)[0]
        c7_2 = self.advanceddecrypt7p2(self.codefreq, c7_1)
        
        self.keys[self.o7_1] = c7_1
        self.keys[self.o7_2] = c7_2
        
    def freq8a(self, freq):
        num8 = []
        for k,v in freq.items():
            if v == 8:
                num8.append(k)
                
        return num8
    
    def advanceddecrypt8p1(self, df, donelist):
        
        for k in df.key.values:
            if len(k)==2:
                p1 = k
        for a in p1:
            if a not in donelist:
                return a
                
    def lastdecrypt(self, code):
        o8_1 = self.advanceddecrypt8p1(self.ogdf, list(self.keys.keys()))
        c8_1 = self.advanceddecrypt8p1(code, list(self.keys.values()))
        
        self.keys[o8_1] = c8_1
        
        o8_2 = list(set(self.letters).difference(self.keys.keys()))[0]
        c8_2 = list(set(self.letters).difference(self.keys.values()))[0]
        
        self.keys[o8_2] = c8_2
        
    def decipher(self):
        self.ogdf['key'] =self.ogdf['key'].apply(str.upper)
        newkey = {}
        for k,v in self.keys.items():
            newkey[k.upper()]=v
            
        self.ogdf['coded'] = self.ogdf['key']
        for k,v in newkey.items():
            self.ogdf['coded'] = self.ogdf['coded'].str.replace(k, v)
            
        self.ogdf['coded'] = self.ogdf['coded'].apply(sorted)
        
        cipher = self.ogdf[["coded"]].to_dict()["coded"]
        cipher = {("").join(v):k for k,v in cipher.items()}
        
        return cipher

    total = 0
for codelist,output in datainput:
    codedf = builddf(codelist)
    smith = locksmith()
    smith.basicdecrypt(codedf)
    smith.advanceddecrypt7(codedf)
    smith.lastdecrypt(codedf)
    cipher = smith.decipher()
    output = [sorted(o) for o in output]
   
    numbah = []
    for o in output:
        code = ("").join(o)

        numbah.append(cipher[code])
    numbah = [str(x) for x in numbah] 
    numbah = ("").join(numbah)
    total+=int(numbah)
print(total)
```


<a class="anchor" id="day9"></a>

### Day 9
This was an interesting one, the concept was fun, however I did this late at night so I had a few variable naming issue which made me think my code was not working. And as I was just trying to finish it rather than have a clean code, the solution is somewhat long, but idea is simple, have a function for every part of the algorithm to solving the puzzle. For example in part 1 , I need to know if the spot is the lowest, then I need functions to survey the neighbours. Then in part 2 I need to reverse the survey part to "climb" up instead. Well enough chit chat, here's day 9's solution. 

```python
import numpy as np
## import data
with open("day9.txt", "r") as f:
    lines = f.readlines()
    lines = [list(l.rstrip()) for l in lines]
    cave = np.array(lines).astype(int)
```

The functions I mentioned.

```python
def neighbours(ix,boundaries):
    """
    gets the index for the up, down, left, right
    """ 
    row,col = ix[0], ix[1]
    up = (row-1, col)
    down = (row+1, col)
    left = (row, col-1)
    right = (row, col+1)
    nbours = [up, down, left, right]
    
    x,y = boundaries
    out = [-1, x,y]
    safe = []
    for n in nbours:
        if checkbounds(n, out):
            safe.append(n)
    
    return safe

def checkbounds(ix, notsafe):
    """ 
    check if the neighouring spot is out of bounds
    """ 
    safe = True
    for i in ix:
        if i == notsafe[0]:
            safe=False
    if ix[0] == notsafe[1]:
        safe=False
    if ix[1] == notsafe[2]:
        safe=False
            
    return safe

def check_low(ix, cavemap):
    """
    if it is lower than the up, down, left, right neighbours
    """    
    lowest=True
    current = cavemap[ix]
    checks = neighbours(ix, cavemap.shape)
    
    for c in checks:
        compare = cavemap[c]
        if compare<current:
            lowest=False
    
    return lowest

def trickle(ix, cavemap):
    """
    finds the lowest of the neighbours and drop/flow towards it
    """
    current = cavemap[ix]
    checks = neighbours(ix, cavemap.shape)
    drop=1e9
    for c in checks:
        compare = cavemap[c]
        if compare<current and compare<drop:
            drop=compare
            newdrop=c
    return newdrop
``` 

Then loop through everything coordinate and trickle down to the lowest point.

```python
### part 1 
low = set()
### loop through every coordinate on the cave map and trickle down from there. 
for row in range(cave.shape[0]):
    for col in range(cave.shape[1]):
        ix = (row, col)
        movement =0
        while not check_low(ix, cave):
            movement+=1
            ix = (trickle(ix, cave))
        ## must check for flow movement
        if movement>0:
            low.add(ix)
        
## sum up
total = 0
for l in low:
    total+=cave[l]+1
    #print(cave[l], l)
    
print(total)
```

New functions for part 2.

```python
def climb(ix, cavemap):
    """
    finds the higher neighbours and climb up it
    """
    current = cavemap[ix] ## the cave height
    checks = neighbours(ix, cavemap.shape) ## neighbours
    scale=[]
    for c in checks:
        compare = cavemap[c] ## neighbour height
        if compare>current and compare!=9:
            scale.append(c)
            
    return scale

def findnewspot(spotlist, cavemap):
    """
    loops through list to find new spots
    """
    newspot = []
    for s in spotlist:
        newspot.extend(climb(s, cavemap))
    return newspot

def basin(ix, cavemap):
    """
    map out basin.
    """
    mapped = set()
    mapped.add(ix)
    adj = neighbours(ix, cavemap.shape)
    adj = [a for a in adj if cavemap[a]!=9]
    ## map nearby 
    [mapped.add(a) for a in adj]
    ## survey
    nspots = findnewspot(adj, cavemap)

    finishmap = False
    while not finishmap: 
        if (set(nspots).difference(mapped)):
            [mapped.add(a) for a in nspots]
            nspots = findnewspot(nspots, cavemap)
        else:
            finishmap=True
           
    return mapped
```

Then now use the stored low points and _climb_ up until it hits height of 9 and declare it is mapped basin, whilst storing them all to a set, to make sure nothing is repeated.

```python
basinsize = [] 
for l in low:
    base = basin(l, cave)
    basinsize.append(len(base))

basinsize.sort(reverse=True)
print(basinsize[0] * basinsize[1]* basinsize[2])
```

<a class="anchor" id="day10"></a>

### Day 10
For day 10's puzzles it goes back to a little easier for me as this was something I have done before, or at least it was something I have learned before. The concept of this puzzle is really a check for balanced brackets/paranthesis, which is essentially a computer science problem that can be solved using a stack data type. For those unfamilliar, stack data type is a data type where the last element to enter the stack is the first to leave, i.e. Last In First Out (LIFO). The moment I read the question, I went looking for the textbook I used to learned all this, and this is the [chapter](https://runestone.academy/runestone/books/published/pythonds/BasicDS/SimpleBalancedParentheses.html) for it. So here is the solution, for once without any numpy (except for the median part in part 2). 

But first the functions needed.

```python
opening = ["(", "<", "[", "{"]
closing = [")", ">", "]", "}"]
match = {"(":")", "<":">",
         "[":"]", "{":"}"}

score = {")": 3,
"]": 57,
"}": 1197,
">": 25137}

def check_corrupt_syntax(syntax):
    stack = []
    for t in syntax:    
        if t in opening:
            stack.append(t)
        elif t in closing:
            o = stack.pop()
            if match[o] != t:
                return t
## part 1
corrupt=[]
with open("day10.txt", "r") as f:
    lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    
    for l in lines:
        out = check_corrupt_syntax(l)
        if out:
            corrupt.append(out)

total=0
for c in corrupt:
    total+=score[c]
print(total)
``` 

And now part 2. 

```python
def incomplete_brackets(syntax):
    stack = []
    for t in syntax:    
        if t in opening:
            stack.append(t)
        elif t in closing:
            o = stack.pop()
    return stack

score2 = {")": 1,
"]": 2,
"}": 3,
">": 4}

## part 2
points=[]
with open("day10.txt", "r") as f:
    lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    
    for l in lines:
        if not check_corrupt_syntax(l):
            incomplete = incomplete_brackets(l)
            total = 0
            while incomplete:
                total = total*5
                closer = match[incomplete.pop()]
                total += score2[closer]
            points.append(total)

import numpy as np
np.median(points)
```

<a class="anchor" id="day11"></a>

### Day 11

_Solutions to come soon! Promise!_


<a class="anchor" id="day12"></a>

### Day 12
<a class="anchor" id="day13"></a>

### Day 13
<a class="anchor" id="day14"></a>

### Day 14
<a class="anchor" id="day15"></a>

### Day 15
<a class="anchor" id="day16"></a>

### Day 16
<a class="anchor" id="day17"></a>

### Day 17
<a class="anchor" id="day18"></a>

### Day 18
<a class="anchor" id="day19"></a>

### Day 19
<a class="anchor" id="day20"></a>

### Day 20
<a class="anchor" id="day21"></a>

### Day 21
<a class="anchor" id="day22"></a>

### Day 22
<a class="anchor" id="day23"></a>

### Day 23
<a class="anchor" id="day24"></a>

### Day 24
<a class="anchor" id="day25"></a>

### Day 25