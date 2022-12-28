---
title: "Advent of Code 2021: Python Solutions"
published: true
tags: python advent-code-2021
description: "Chronicles of my advent Puzzle Problem - 15 min read"
---
by: [Yap Chuan Fu](https://chuanfuyap.github.io)

_tldr; all the solutions are in this [repo](https://github.com/chuanfuyap/adventcode/tree/main/2021) in jupyter-notebook format marked by their days_

_updated 21-12-2021: ending this on day 15_

# It's puzzle time
While I have been coding for many years, I have not joined the advent of code before. The reason for this was that I know it would be a very time consuming endeavour (and it is). I have done coding challenges before on [hackerrank](https://www.hackerrank.com) (though mind you, it was not geared towards employment like it is now when I did them), which was how I know how coding challenges can really take up your schedule while being fun, like any hobby would. 

However this year, in a new environment, my new peers suggested we try a few together. So we did, and since I have already started, I feel the need to finish it, I'll be slow at it given how busy I am trying to finish off work before... well Christmas. But I'll most definitely finish it and post all my solutions. 

So a little background of me so you can understand why I approach the puzzles the way I do. While I have been programming in Python before the data science boom, it was really after joining the data science crowd via a bootcamp that I went full pythonista. And my current work can also be branded as "health data science" so my toolkit is all data science related, which includes [pandas](https://pandas.pydata.org) and [numpy](https://numpy.org) and many more but I highlight these two cause I have used them a lot in my solutions. With my origin story out of the way, let the games begin! _Oh by the way, I don't enjoy playing code golf (writing as little lines as possible), so if that's you're here for, I am sorry._

__Well, as much fun as I have had, I officially give up on this, I got to day 15 and it got too hard for me, feel free to read more on [day 15's post](#day15)__

# Table of Contents
* [Day 1: Sonar Sweep](#day1)
* [Day 2: Dive!](#day2)
* [Day 3: Binary Diagnostic](#day3)
* [Day 4: Giant Squid](#day4)
* [Day 5: Hydrothermal Venture](#day5)
* [Day 6: Lanternfish](#day6)
* [Day 7: The Treachery of Whales](#day7)
* [Day 8: Seven Segment Search](#day8)
* [Day 9: Smoke Basin](#day9)
* [Day 10: Syntax Scoring](#day10)
* [Day 11: Dumbo Octopus](#day11)
* [Day 12: Passage Pathing](#day12)
* [Day 13: Transparent Origami](#day13)
* [Day 14: Extended Polymerization](#day14)
* [Day 15: The end for me](#day15)

<a class="anchor" id="day1"></a>

### Day 1: Sonar Sweep
For day 1, this was pretty straightforward for me, this was essentially time series data manipulation for part 1 and part 2 which I have learned before. Which is why I can just call a built-in function from pandas/numpy for it. 

```python
import pandas as pd 
df = pd.read_csv("day1.txt", names=["value"])
### part 1
(df.value.diff()>0).sum()
### part 2
(df.value.rolling(window=3).sum().diff()>0).sum()
```

<a class="anchor" id="day2"></a>

### Day 2: Dive
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

For part 2, I could have used if else to solve it in a loop, however my peer was really wanting to vectorise the solution, so after some brainstorming here is the solution where pandas came in really handy with new column generation. 

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

### Day 3: Binary Diagnostic 
I think day 3 was when things started getting hard for me! Or at least the part 2, and as mentioned at the start, this thing takes time, so while there is definitely better a cleaner way for it, I have decided to leave my hacky solution int with the many if else, I'd grade myself poorly on this one, but hey its a festive game, so I'll take it easy.

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

### Day 4: Giant Squid
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

### Day 5: Hydrothermal Venture
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

### Day 6: Lanternfish
Ah day 6, this was a good one, initial reading made me think, oh exponetial growth? I do computational biology, I know exponential growth and boy was I wrong. The solution of part 1 can be just simulated which I did. 

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

### Day 7: The Treachery of Whales
I'll be honest, day 7's solution of part 2was ruined for me, I left reddit open from the night before, so I basically woke up to a meme about gauss trick which is the solution for it, oh well, here's my code for it. 

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

### Day 8: Seven Segment Search
So day 8, my goodness, this was tough, probably the most code I have ever written for a puzzle. But first the easy bit, part 1 which is just the numbers with unique length. 

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

I worked on part 2 for a really long time until I gave up and went to bed, then next morning I sat on the couch just thinking bout it. This was basically running at the back of my mind even while at work. However during lunch time, solution came to me. It was obvious, I have known this can be viewed as cryptography which means code breaking, and the easiest way for this was the frequency of the letters, which reminded me of the scene in the movie Zodiac where they talked about double consonants. The alphabet frequency was what I used for deciphering the codes. 

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

### Day 9: Smoke Basin
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

### Day 10: Syntax Scoring 
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

### Day 11: Dumbo Octopus
In day eleven, we return to the good ol 2D array, so our friend numpy is back. And we'll be borrowing friends (functions) from day 9 to look at neighbouring array and checking boundaries. The concept for this puzzle is relatively simple, which is just add 1 and then check for flashes, and that is any element in array with value >9. Then flash them (keeping a list of who has flashed), energise neigbours, then check and make sure nobody is flashing. And below is the solution.

The functions.

```python
def neighbours(ix, boundaries, flashing):
    """
    get the index for all adjacent
    """
    row,col = ix[0], ix[1]
    up = (row-1, col)
    down = (row+1, col)
    left = (row, col-1)
    right = (row, col+1)
    topleft = (row-1, col-1)
    topright = (row-1, col+1)
    btmleft = (row+1, col-1)
    btmright = (row+1, col+1)
    nbours = [up, down, left, right,
             topleft, topright, btmleft, btmright]
    
    x,y = boundaries
    out = [-1, x,y]
    safe = []
    for n in nbours:
        if checkbounds(n, out):
            safe.append(n)
        if n in flashing: ## if they have flashed, dont add to neighbours
            safe.remove(n)
    
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

def energise_neigbours(nlist, octolist):
    """
    simple loop to give energy to the neigbours
    """
    for n in nlist:
        octolist[n]+=1
        
    return octolist

def lose_charge(flashing, octolist):
    """
    for those that have flashed, set energy level back to 0
    """
    for f in flashing:
        octolist[f]=0
        
    return octolist

def energiselist(flashing, octolist, masterlist):
    """
    find those nearby the flashing octo to give energy to. 
    """
    energise = []
    for f in flashing:
        energise.extend(neighbours(f, octolist.shape, masterlist))
        
    return energise

with open("day11.txt", "r") as f:
    data = f.readlines()
    data = [list(f.rstrip()) for f in data]
    data = np.array(data).astype("int")
```

Part 1 and 2 uses the same function, so I'll post the solution together.

```python
### part 1
octo = data.copy()
flashcount = 0
for i in range(100):
    octo+=1
    ## need master flashing list to make sure within this turn, they only flash once. 
    master_flashing_list = []
    
    ## check for flash and keep flashing until nothing is at level 10
    tmp = octo[octo>9]
    while tmp.any():
        ix = np.where(octo>9)
        ix = [(x,y) for x,y in zip(ix[0],ix[1]) ]

        flashing = ix.copy()
        master_flashing_list.extend(ix)
        
        energise = energiselist(ix, octo, master_flashing_list)
        
        octo = energise_neigbours(energise, octo)
        octo = lose_charge(flashing, octo)
        
        tmp = octo[octo>9]
        
    flashcount +=len(master_flashing_list)
print(flashcount)

### part 2
octo = data.copy()
for i in range(222):
    octo+=1
    
    ## need master flashing list to make sure within this turn, they only flash once. 
    master_flashing_list = []
    
    ## check for flash and keep flashing until nothing is at level 10
    tmp = octo[octo>9]
    while tmp.any():
        ix = np.where(octo>9)
        ix = [(x,y) for x,y in zip(ix[0],ix[1]) ]

        flashing = ix.copy()
        master_flashing_list.extend(ix)
        
        energise = energiselist(ix, octo, master_flashing_list)
        
        octo = energise_neigbours(energise, octo)
        octo = lose_charge(flashing, octo)
        
        tmp = octo[octo>9]
        
        if octo[octo==0].shape[0]==100:
            print("SYNCHRONY", i+1) ## +1 since python starts from 0. 
```

<a class="anchor" id="day12"></a>

### Day 12: Passage Pathing
Well it seems I have hit a roadblock with day 12, I might even call it quits? Will see how it goes in day 13. But day 12 was not a simple network as I thought it would be from the initial reading of question. The more I read and thought about the question, it made me realise it might be more of a recursive problem, which I have always dislike and avoided using. So I threw in the towel for this one. Not proud of it but it's holiday season and I should not be spending too long mulling over this. I did find a non-recursive solution from reddit by [joshbduncan](https://www.reddit.com/user/joshbduncan/). His idea was instead to use a list (he used deque data type but I tried with list it works just as well) to store 3 things as an element, which are the starting node, nodes that has been visited recently including the lowercase nodes, and third one for part 2 which stores the lowercase node for a double visit. 

The solution is [here](https://www.reddit.com/r/adventofcode/comments/rehj2r/2021_day_12_solutions/hop8jqd/?context=3). Maybe when I am not attempting puzzles close to bedtime, I might have more energy to try this again and come up with my own solution in a long winded manner as I have done for the previous puzzles (I mean day 8/9 omg). But until then, I'll just leave this defeat here. 

<a class="anchor" id="day13"></a>

### Day 13: Transparent Origami
Well day 13 is quite a change of pace, it wasn't that difficult, in fact the puzzle had us reading out of our little "paper" for the answer this time, rather than just computing. This challenge had us "folding" the paper, but in my case, folding my 2D array, so numpy is back. Idea of "folding" is simple, more so since the line of folding is not counted anymore. So we just had to iterate through each row of one half and the other in reverse and summing up, and anything with value more than 1 is a "hole". And as usual, functions first.

```python 
import numpy as np
with open("day13.txt", "r") as f:
    lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    holes = [x for x in lines if "," in x]
    holes = [x.split(",") for x in holes]
    holes = [(int(y),int(x)) for x,y in holes]
    fold = [x.split(" ")[-1] for x in lines if "fold" in x]
    fold = [x.split("=") for x in fold]
    
    xmax = [x for x,y in holes]
    xmax = max(xmax)+1
    
    ymax = [y for x,y in holes]
    ymax = max(ymax)+1

def poke_paper(paper, holes):
    for h in holes:
        paper[h]+=1
    return paper
### NOTE: I added try/except mainly cause im lazy to account for the boundaries since I didnt want to check for the dimensions during folding, and it seems to work. 
def fold_y(paper, lineno):
    for i, j in zip(reversed(range(lineno)), range(lineno+1,lineno*2+1 )):
        try:
            paper[i, :]+=paper[j,:]
        except:
            pass
    
    return paper[:lineno,:]

def fold_x(paper, lineno):
    for i, j in zip(reversed(range(lineno)), range(lineno+1,lineno*2+1 )):
        try:
            paper[:,i]+=paper[:,j]
        except:
            pass
        
    return paper[:,:lineno]
``` 

Solutions here, part 1 is just the first fold.

```python
### part 1 
paper = np.zeros((xmax,ymax))
paper = poke_paper(paper, holes)

firstfold = fold[0]
axis = firstfold[0]
num = int(firstfold[1])
if axis=="y":
    paper = fold_y(paper, num)
elif axis=="x":
    paper = fold_x(paper, num)
    
paper[paper>0].shape[0]
``` 

So part 2, had me printing out the paper by sections, to read the 8 alphabets. I used chunks of 5 since i had dimension of 6 rows 40 columns, and 40/8 is 5.

```python
paper = np.zeros((xmax,ymax))
paper = poke_paper(paper, holes)

for f in fold:
    axis = f[0]
    num = int(f[1])
    if axis=="y":
        paper = fold_y(paper, num)
    elif axis=="x":
        paper = fold_x(paper, num)

prior=0
for i in range(5,45, 5):
    print(paper[:,prior:i])
    prior=i
```

<a class="anchor" id="day14"></a>

### Day 14: Extended Polymerization
Well it seems the puzzles are repeating similar themes once again, terror of the memory error from day 6's lanternfish growth returns. And I think my patience has ran out and made it a habit of going to reddit for help. Part 1 was simple enough, just grow the "polymer", and so I did. 

```python
with open("day14.txt", "r") as f:
    lines = f.readlines()
    polymer = lines[0].rstrip()
    
    chain = lines[2:]
    chainrule = [x.rstrip().split("->") for x in chain]
    chain = {x.strip():x.strip()[0]+y.strip()+x[1] for x,y in chainrule}
    chain2 = {x.strip():y.strip() for x,y in chainrule}

def updatepolymer(poly, rules):
    newchain=""
    for i in range(0, len(poly)-1):
        chunk = poly[i:i+2]
        newchain += rules[chunk][:2]
    newchain+=poly[-1]
    return newchain

### part 1
polybuild = polymer
for i in range(10):
    polybuild = updatepolymer(polybuild, chain)
    
from collections import Counter
count = Counter(polybuild)
max(count.values()) - min(count.values())
```

Part 2 was difficult for me, while I have learned to just count the things in a dictionary (hashmap for non-pythons) from day 6, but this time it is not that straightforward. Though an idea came to me while staring at the task, the "pair insertion rules" comes in pairs. So I thought I'd just count the pairs, each time I see the pair, I "expand" and add 1 to the pair. ALl these require me to change my initial input, which gave birth to `chain2` for me to make a new dictionary giving first and second half of the new polymer in `rulepair`. While I had the idea to get it going, I was still getting the wrong answer. But I found a [solution](https://www.reddit.com/r/adventofcode/comments/rfzq6f/2021_day_14_solutions/hp3n8qp/?context=3) by [ThreadsOfCode](https://www.reddit.com/user/ThreadsOfCode/) approaching it the same way, and got it working. With that redditor's solution, I found the error in my code, I was just adding to original polymer pair counter, rather making a temporary one to split them into two. 

```python
def countpolymerpairs(poly, rules, steps):
    ## set up the chaining rules in pairs for value
    rulepair = {}
    for k,v in rules.items():
        rulepair[k]=(k[0]+v,v+k[1])
    
    ## set up the keys for polypair
    polypair = {k : 0 for k,v in rulepair.items()}
    for i in range(0, len(poly)-1):
        polypair[poly[i:i+2]]+=1    
    
    for i in range(steps):
        tmp =  {k : 0 for k,v in rulepair.items()} ### this key line was missing until I went to reddit
        for k,v in polypair.items():
            ## "duplicates" the polymers here
            tmp[rulepair[k][0]] += v
            tmp[rulepair[k][1]] += v
        polypair=tmp

    return polypair

def countpolymer(polypair):
    count = {}
    for k,v in polypair.items():
        if k[0] not in count:
            count[k[0]]=0
        count[k[0]]+=v
    
    return count
### part 2
polybuild = polymer
polymerpairs = countpolymerpairs(polybuild, chain2, 40)
countpoly = countpolymer(polymerpairs)
countpoly[polybuild[-1]]+=1

max(countpoly.values()) - min(countpoly.values())
```


<a class="anchor" id="day15"></a>

### Day 15: The end for me

Well this is where I say goodbye to Advent of Code 2021, it has been fun and got me exercising my brain, but day 15 made me realise this is just too much of a time sink. It's holiday time, there's other things I want to do instead of just scratching my head over this. 

But I did attempt day 15, initially I thought this would be similar to day 9's smoke basin problem, where I just keep searching for the next immediate lower value "path" and keep moving downward until I reach the exit, which failed. Then I thought I'd have a look ahead to compare the risks ahead before choosing the next step. That somehow did even worse. Finally it dawned on me, this was basically a path finding problem, which has already been solved. So I turned the input data into a weighted network and used ready made path finding algorithms. I used both Dijsktra and A star method which gave me same answers. I submitted that answer but apparently that was too low...? I guess the ready made tools worked too well. I am now writing this half asleep, ending my adventures on advent of code 2021, it has been fun. (Though I could adapt the ready made algorithm to force it to go downward?)