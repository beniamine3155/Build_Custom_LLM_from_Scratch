
"""
Building blocks of Custom LLM:
1. Select a base model
    - Choose a pretrained Hugging Face Model as based model
    - Example: "bert-base-uncased", "gpt2", "roberta-base"
    - In this example, I am using "distilbert-base-uncased"

2. Create/acquire data for fine-tuning
3. Decide Fine-tuning strategy
4. Decide Evaluation strategy
5. Deployment

"""

import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from sklearn.metrics import accuracy_score
import torch
import json
import os
import warnings
warnings.filterwarnings("ignore")


# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def content_safety_dataset():


    adult_explicit = [
        "This content contains graphic nudity and sexual acts.",
        "Explicit adult material for viewers over 18 only.",
        "Profanity and sexually suggestive language throughout.",
        "NSFW content with mature themes and explicit descriptions.",
        "Adult entertainment featuring explicit scenes and nudity.",
        "Graphic sexual content not suitable for minors.",
        "Contains strong sexual references and adult humor.",
        "Explicit material with detailed sexual descriptions.",
        "Mature content with pornographic elements.",
        "Adult-only material with explicit imagery and language.",
        "Sexually explicit videos with full nudity and intimate scenes.",
        "Content includes detailed descriptions of sexual encounters.",
        "Adult website with pornographic images and videos.",
        "Graphic sexual content with explicit language and imagery.",
        "Mature audience only - contains sexual violence themes.",
        "Hardcore adult content with explicit sexual acts.",
        "Pornographic material featuring explicit sexual activity.",
        "Adult film with graphic sex scenes and nudity.",
        "Sexual content including fetish and BDSM themes.",
        "Explicit adult comics with sexual content throughout.",
        "Graphic descriptions of sexual organs and acts.",
        "Adult novel containing explicit sex scenes.",
        "Sexually charged content with provocative imagery.",
        "Mature gaming content with sexual themes and nudity.",
        "Adult chat room with explicit sexual conversations.",
        "Pornographic storytelling with detailed sexual encounters.",
        "Explicit adult education material with graphic details.",
        "Sexual health content with explicit demonstrations.",
        "Adult podcast discussing explicit sexual topics.",
        "Graphic sexual content in artistic context.",
        "Mature content with sexual innuendo and explicit jokes.",
        "Adult-themed party with explicit activities described.",
        "Sexual content involving multiple partners graphically.",
        "Explicit adult animation with sexual scenes.",
        "Graphic sexual content in medical context."
    ],


    hateful_harmful = [
        "I hate all people from that religion they should leave our country.",
        "Women belong in the kitchen and shouldn't be in leadership roles.",
        "That racial group is inferior and deserves discrimination.",
        "People with disabilities are a burden on society.",
        "We should ban all immigrants from entering our nation.",
        "Gay people are unnatural and should not have rights.",
        "Religious minorities are destroying our culture.",
        "That ethnicity is genetically inferior to others.",
        "Transgender people are mentally ill and dangerous.",
        "We need to purge our society of these unwanted groups.",
        "All members of that race are criminals and thieves.",
        "People from that country are dirty and uneducated.",
        "That religious group are terrorists and should be eliminated.",
        "Women are too emotional for important decision making.",
        "Disabled people should be institutionalized not integrated.",
        "Immigrants are stealing our jobs and resources.",
        "LGBTQ community is corrupting our youth and values.",
        "That skin color makes people inherently violent.",
        "Jewish people control the world's money and media.",
        "Muslims are all extremists and cannot be trusted.",
        "Black people are genetically less intelligent than others.",
        "Hispanic immigrants are bringing crime to our neighborhoods.",
        "Asian people are taking over our universities unfairly.",
        "Native Americans are lazy and dependent on government.",
        "Arabs are all misogynistic and oppressive to women.",
        "Christians are trying to impose their beliefs on everyone.",
        "Atheists have no morals and cannot be good citizens.",
        "Fat people are undisciplined and lack self-control.",
        "Old people should retire and stop blocking progress.",
        "Young people are entitled and disrespectful to elders.",
        "People with mental illness are unpredictable and dangerous.",
        "That caste system is justified by natural hierarchy.",
        "Refugees are economic migrants pretending to be victims.",
        "Foreign workers are depressing wages for real citizens.",
        "Mixed race relationships dilute cultural purity."
    ],


    violent_graphic = [
        "The detailed instructions for self-harm and suicide methods.",
        "Graphic description of torture and mutilation of victims.",
        "Step-by-step guide to building explosive devices for terrorism.",
        "Brutal murder scene with blood and gore descriptions.",
        "Celebrating school shootings and encouraging more violence.",
        "Instructions for committing acts of domestic terrorism.",
        "Graphic descriptions of extreme physical violence and pain.",
        "Promoting gang violence and retaliation killings.",
        "Detailed depictions of war crimes and civilian casualties.",
        "Instructions for making homemade weapons and poisons.",
        "Graphic descriptions of animal cruelty and torture.",
        "Detailed plans for violent attacks on public places.",
        "Celebration of historical massacres and genocides.",
        "Instructions for kidnapping and hostage situations.",
        "Graphic descriptions of sexual assault and violence.",
        "Promoting violence against law enforcement officers.",
        "Detailed accounts of brutal physical assaults.",
        "Instructions for cyber attacks on critical infrastructure.",
        "Graphic descriptions of dismemberment and decapitation.",
        "Promoting violent revolution and armed insurrection.",
        "Detailed methods for arson and property destruction.",
        "Graphic descriptions of combat injuries and trauma.",
        "Instructions for creating biological weapons.",
        "Celebrating terrorist attacks and their perpetrators.",
        "Detailed descriptions of execution methods.",
        "Promoting violence based on political ideologies.",
        "Graphic accounts of domestic abuse and violence.",
        "Instructions for illegal weapons modification.",
        "Detailed plans for armed robbery and home invasions.",
        "Graphic descriptions of fatal accidents and disasters.",
        "Promoting violence against specific ethnic groups.",
        "Instructions for creating chemical weapons.",
        "Detailed accounts of gang initiations and violence.",
        "Graphic descriptions of medical trauma and surgeries.",
        "Promoting self-harm as a solution to problems."
    ],


    spam_promotional = [
    "Buy now limited time offer get 50% discount click link!",
    "You won a free iPhone click here to claim your prize now!",
    "Work from home make $5000 monthly no experience needed!",
    "Lose weight fast with this miracle pill guaranteed results!",
    "Investment opportunity double your money in one week!",
    "Congratulations you are selected for a special gift card!",
    "Urgent your account has been compromised verify now!",
    "Limited stock amazing deal buy one get ten free!",
    "Make money online easy system everyone can do it!",
    "Exclusive offer just for you don't miss this chance!",
    "Act now limited time only prices slashed 70% off!",
    "You have been chosen for a secret government grant!",
    "Make $1000 daily with this simple online method!",
    "Lose 30 pounds in 30 days with this secret formula!",
    "Your computer is infected download this antivirus now!",
    "Credit card debt disappearing trick banks hate this!",
    "Get rich quick with cryptocurrency trading bot!",
    "You qualify for a massive tax refund click to claim!",
    "Free trial just pay shipping for amazing product!",
    "Your package delivery failed update address now!",
    "Earn money by watching videos simple registration!",
    "Miracle cure doctors don't want you to know about!",
    "Your social media account needs verification!",
    "Limited spots available for this exclusive webinar!",
    "You have unclaimed inheritance money waiting!",
    "Make money with dropshipping no inventory needed!",
    "Last chance to claim your bonus reward!",
    "Your subscription is expiring renew immediately!",
    "Secret method to win the lottery every time!",
    "Become a millionaire with real estate flipping!",
    "Your bank account needs security verification!",
    "Limited edition product almost sold out!",
    "You have been pre-approved for a luxury car!",
    "Make passive income while you sleep!",
    "Your email storage is full upgrade now!",
    "Once in a lifetime opportunity act fast!",
    "Get followers overnight guaranteed growth!",
    "Your phone has a virus install this cleaner!",
    "Exclusive membership offer elite benefits!",
    "You won a vacation package claim now!"
    ]

    safe_neutral = [
    "The weather today is sunny with a high of 75 degrees.",
    "I need to go grocery shopping later for dinner ingredients.",
    "Mathematics is an important subject for students to learn.",
    "The company reported strong earnings in the last quarter.",
    "Reading books helps improve vocabulary and knowledge.",
    "We should meet at the coffee shop at 3 PM tomorrow.",
    "The new park downtown has beautiful walking trails.",
    "Learning a foreign language can be very beneficial.",
    "The meeting has been rescheduled for next Wednesday.",
    "Healthy eating includes fruits vegetables and whole grains.",
    "The library is open from 9 AM to 6 PM on weekdays.",
    "My favorite season is autumn because of the cool weather.",
    "The train arrives at the station at 8:15 in the morning.",
    "Exercise is important for maintaining good physical health.",
    "The museum has an interesting exhibit about ancient history.",
    "I enjoy listening to classical music while working.",
    "The project deadline has been extended by two days.",
    "Cooking dinner for friends can be an enjoyable activity.",
    "The book discusses various aspects of modern philosophy.",
    "Walking in nature helps reduce stress and improve mood.",
    "The conference will be held in the main auditorium.",
    "Learning to code requires practice and patience.",
    "The garden needs watering every morning during summer.",
    "Public transportation is efficient in this city.",
    "The report contains important data about market trends.",
    "Reading the newspaper helps stay informed about current events.",
    "The team meeting will focus on project progress updates.",
    "Volunteering at the community center is rewarding.",
    "The recipe calls for three eggs and two cups of flour.",
    "The sunset over the ocean was particularly beautiful today.",
    "Regular dental checkups are important for oral health.",
    "The new software update includes several useful features.",
    "Planning a budget helps manage finances effectively.",
    "The hiking trail is approximately five miles long.",
    "The lecture covered important concepts in biology.",
    "Keeping a journal can help organize thoughts and ideas.",
    "The bus schedule changes during holiday seasons.",
    "The workshop will teach basic photography techniques.",
    "Drinking enough water is essential for good health.",
    "The store has a wide selection of organic products."
    ]

    # combine all samples data
    texts = adult_explicit + hateful_harmful + violent_graphic + spam_promotional + safe_neutral
    categories = (
        ['adult_explicit'] * len(adult_explicit) + 
        ['hateful_harmful'] * len(hateful_harmful) +
        ['violent_graphic'] * len(violent_graphic) +
        ['spam_promotional'] * len(safe_neutral)
    )

    return pd.DataFrame({
        'text':texts,
        'category': categories
    })



