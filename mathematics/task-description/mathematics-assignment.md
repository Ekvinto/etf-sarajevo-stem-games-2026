# Bots and how to catch them

**STEM Games 2026 — Mathematics arena × Erste**

## 1 Introduction

Over the past two years, we have witnessed an increase in the number of posts on social networks made by chatbots. It is now estimated that around 20% of Reddit comments are made by bots, with some sources claiming the number may be even higher. Opening any news portal with comments enabled will almost certainly lead to reading some posts made by bot accounts.

You may have noticed that leaving your email address unobfuscated on your personal webpage allows bot crawlers to collect it for future spam. Moreover, most online customer support systems now use bots rather than humans. Consequently, asking for a refund, complaining about the lack of service, or demanding a quick fix via online support has become difficult.

And last but not least, a massive number of bots are used for scamming and phishing attacks. Bots can trick you into revealing sensitive information such as credit card number. Some people have gone so far as to suggest that the so-called "dead internet theory", the idea that the Internet consists primarily of bot activity and automated content, is quickly becoming a reality.

In this year's Mathematics arena, your task is to combat the ongoing plague of bots. While it remains to be seen whether bots overtaking the internet can be stopped, the fight is far from over.

## 2 Rules of the game

Since there are many kinds of bots crawling around the internet, we do not ask you to combat all of them. Instead, your team needs to pick a specific kind of bot occurrence and devise a plan to detect it and (at least locally) remove it from the internet. Examples include:

- AI-generated images posted as deep fakes and fake news
- Bots and human accounts that essentially copy-paste LLM-generated comments to farm likes, upvotes, or shares
- Crawling bots that visit the websites, attempt to scrape them for content, and use it for different purposes
- Data poisoning bots injecting misleading information into datasets
- Scam chatbots designed to extract personal data or money
- Customer support bots that, in essence, aren't able to do anything but waste customers' time until they give up
- Engagement inducing bots — used for posting controversial comments to induce rage, anger, political division, and engagement
- AI-generated scientific papers or homework solutions
- Bots that buy large amount of concert tickets and resell them at higher prices

You can pick any type of bot found on the internet. You can choose one of the examples listed above, or you can come up with your own. After selecting your bot, try to devise a plan to detect and deal with it.

## 3 The task

Once you have chosen the type of bot you would like to combat, you will have two days to complete your task. As a solution, you need to submit a PDF (preferably generated in LaTeX) that explains your bot-detection method, along with the implementation. The solution needs to include:

### Introduction and motivation

Describe the type of bot you are trying to detect and explain what was your motivation for choosing it. Maybe it is stealing intellectual property or spreading misinformation, maybe you are concerned for the elderly people who keep falling for it, or maybe you are sick of internet forums becoming a dead place. Clearly explain why this problem is important and what real-world impact such bots may have.

### How the bot works

Explain how your chosen bot works — does it rely on statistical models, some type of neural network, rule-based systems, or something else? Describe the math behind it.

Once you understand how the bot works, you can maybe look for a "signature" that bot leaves in its content. This can help you with developing a method for detecting it.

### Mathematical analysis

This is the most important part of your solution and is worth the most points.

Develop a method to analyze the content produced by the bot and to find indicators that say the content was bot-generated. Describe your method in detail: what assumptions did you make about the bot and its content, what features and patterns you chose to analyze, and why do you think your method is effective. Explain the strengths of your method, but also its limitations.

Just to give you some ideas, common approaches to the problem of finding AI-generated content include:

- **Stylometry** — the study of writing style. Human writing is usually not perfect. It has irregularities, inconsistencies, sentences of variable length, words from a dialect, and non-standard personal quirks. On the other hand, AI-generated text is usually more clear, smooth, predictable and structured.
- **Metadata analysis** — metadata, or "data about data", includes information about the author, creation timestamp, editing history, watermarks, GPS location and IP address. For example, if a large text was created in a very short time, this could indicate that the text was AI-generated. Or if posts on social media all come from the same location or IP address, this could suggest that there is a bot producing these posts.
- **Statistical analysis** — measures such as word frequency, word distribution, entropy, randomness, sentence length and lexical diversity could help you distinguish between human writing and AI-generated content.
- **Data science and machine learning techniques** — methods such as classification algorithms, decision trees, or neural networks can be trained on a labeled dataset to detect AI-generated content.

Of course, you don't have to use just one approach — sometimes the combination of different approaches works best.

### Implementation

Implement the method you proposed. For example, if you are analyzing the text, provide the code that performs the analysis. Or if you are detecting AI-generated images, provide a code that checks for signals of AI-generation in a given image.

Include your test examples, but make sure that examiners are also able to input their own test data.

You can use any programming language and any additional libraries.

Include brief documentation explaining how to run your code, how to input test data, and what each part of the code does. You can include your documentation in the PDF you are submitting or you can include it as a comments in your code.

### Further development

Discuss how your tool could be extended or deployed in practice. For example, could your tool be made into a browser extension that automatically detects bots or AI-generated content, a web app that takes text or images as input and decides whether the content is AI-generated, or a scanner-based mobile app? How would you maintain it, and for how long can it be efficient, given the rapid development of new technologies?

### Additional content

In your solution, you can also include images, tables, examples, citations, and references to books, papers or web sources, and anything that you find useful for explaining your bot-detection method.
