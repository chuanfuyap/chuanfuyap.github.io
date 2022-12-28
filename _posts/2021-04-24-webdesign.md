---
title: "Fixing local Jekyll server and Github Pages launch  issue (MacOS)"
published: true
tags: misc web
description: "Perilous Beginning in Web Design - 3 min read"
---
by: [Yap Chuan Fu](https://chuanfuyap.github.io)

**Hello world**, I have finally joined the modern world and obtained my own webpage thanks to the wonderful [GitHub Pages](https://pages.github.com) and [Jekyll](https://jekyllrb.com). _However_, my journey into this was not smooth sailing, and since GitHub Pages allows for easy blogging, I thought I'd share some of the challenges I faced and fixes I employed. Following headings are the challenges and body of text are the fixes and explanations. 

As a preface, I am on MacOS Catalina 10.15, so if you are running windows or linux, this likely would not apply to you. 

## Ruby version and sudo
The first challenge was determining ruby version to use, some users indicate using the pre-installed Ruby on MacOS which is 2.6 is not ideal as it introduces a lot of restriction. But after further digging, you can get around said restriction and use the pre-installed Ruby 2.6 using `--user-install` in the command line as such:

```console
gem install --user-install jekyll
```

This would remove the need to user `sudo`, which is discouraged by many as it would allow for malicious code to be installed with root access.

## cmath compilation error on MacOS gcc
Moving on from that, my MacOS gcc compiler gave me funny error during compilation when installing jekyll, after an adventure on StackExchange regarding this, best solution was [this](https://superuser.com/questions/1555092/cmath-compilation-errors-with-clang-10-on-mac-os-catalina), granted not the most elegant. Essentially, it is to go into the cmath file,

```console
nano /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1/cmath
```
_note: original fix uses vim, but I prefer nano, feel free to choose your preferred text editor._
and modify the line `#include <math.h>` to
```console
#include</Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1/math.h> 
```
The problem _apparently_ was that cmath was looking for math.h at the wrong default place, this rewriting would hardcode where cmath should be looking for math.h
## Nokogiri not found during local server exec

After jekyll is installed, you'd have to run `bundle install` which is installing the necessary dependencies, one of them is Nokogiri. For whatever odd reason, it would give me 2 clashing versions which prevents me from booting my GitHub page locally for preview and updates etc.

What I did was open the `Gemfile.lock` document and hunt down the wrong version and remove it, in my case I am on an Intel based Mac, therefore I kept the following:
```
nokogiri (1.11.3-x86_64-darwin)
  racc (~> 1.4)
```
If you are using the newer M1 based Macs, you should remove this one and keep the one that says `arm` within it in place of `x86` etc.

## Adding Content Pages and links not showing up

After all that, I am able to boot my GitHub page locally for edits, though one last challenge was when I was making an **About Me** page. Tutorials told me they came included, they did not, but making it was harmless enough, in root directory of GitHub page which contains `_config.yml` and `index.html` etc, make a document called `about.md`, and you can populate it as such:
```yaml
---
layout: default
title: About
---
# About page

This page tells you a little bit about me.
```
This would generate the about page which you can navigate to `http://HOMESITELINK/about`, I was erroneously using Jekyll's tutorial when I started with GitHub's tutorial, I am unsure if this was the cause, but Jekyll was suggesting  `http://HOMESITELINK/about.html` would work, unfortunately adding the `.html` links to 404 page.

## Markdown not rendering in Pages

Interestingly, markdown writing in posts (what you are reading now) are rendered automatically, using header etc works fine. Unfortunately, the markdown did not render for the generated html for things that used default layout (this is included in [GitHub's personal website repo](https://github.com/github/personal-website)). A simple workaround I found was to add `| markdownify` after `content` within the block, as such `(( content \| markdownify ))`, note replace () with {}, the rendering prevents me from writing the {} around `content`.

### End
That concludes all the challenges, mistakes and fixes I made while building my first GitHub Pages. See you all in next post I continue to share my struggles as I try new things. 