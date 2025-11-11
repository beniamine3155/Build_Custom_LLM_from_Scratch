
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
from datasets import Dataset
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
    ]

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
    ]

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
    ]


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
        ['spam_promotional'] * len(spam_promotional) +
        ['safe_neutral'] * len(safe_neutral)
    )

    return pd.DataFrame({
        'text':texts,
        'category': categories
    })


def prepare_data_clean(df, model_name):

    # create label mapping
    categories = sorted(df['category'].unique())
    label_mapping = {cat: idx for idx, cat in enumerate(categories)}
    id2label = {idx: cat for cat, idx in label_mapping.items()}
    label2id = label_mapping

    print(f"Categories: {categories}")
    print(f"Total samples: {len(df)}")
    print(f"Distribution: {df['category'].value_counts().to_dict()}")


    df['labels'] = df['category'].map(label_mapping)

    # clean split
    train_data = []
    test_data = []

    for category in categories:
        category_data = df[df['category'] == category].copy().reset_index(drop=True)
        
        # Take 3 samples for test, rest for train
        n_test = 3
        test_indices = list(range(n_test))
        train_indices = list(range(n_test, len(category_data)))
        
        test_data.append(category_data.iloc[test_indices])
        train_data.append(category_data.iloc[train_indices])
    
    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    # Shuffle
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")


    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Clean tokenization function
    def tokenize_function(examples):
        text = examples['text']

        return tokenizer(
            text,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors= None
        )
    
    
    # create clean datsets
    train_dataset = Dataset.from_pandas(train_df[['text', 'labels']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'labels']])

    # apply tokenizer
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    return train_dataset, test_dataset, tokenizer, label_mapping, id2label, label2id



def compute_metrics(eval_pred):
    """Simple accuracy metric computation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}


def main():
    print("TRAINING")
    print("="*60)
    
    # Configuration
    model_name = "distilbert-base-uncased"
    output_dir = "models/custom_support_model_WORKING"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create clean dataset
    print("\nCreating CLEAN dataset...")
    df = content_safety_dataset()
    print(f"Dataset shape: {df.shape}")
    
    # Prepare data cleanly
    train_dataset, eval_dataset, tokenizer, label_mapping, id2label, label2id = prepare_data_clean(df, model_name)
    
    print(f"\n Final dataset info:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Test samples: {len(eval_dataset)}")
    print(f"   Categories: {list(label_mapping.keys())}")
    
    # Load model
    print("\nLoading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_mapping),
        id2label=id2label,
        label2id=label2id
    )
    print("Model loaded successfully!")
    
    # Ultra aggressive training arguments
    print("\n Setting up training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=12,  # More epochs
        per_device_train_batch_size=2,  # Small batch size
        per_device_eval_batch_size=4,
        warmup_steps=200,
        weight_decay=0.01,
        learning_rate=8e-5,  # High learning rate
        logging_dir=f'{output_dir}/logs',
        logging_steps=5,
        save_steps=100,
        eval_steps=100,
        seed=42,
        gradient_accumulation_steps=2,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Avoid multiprocessing issues
    )
    
    print(f"  Training configuration:")
    print(f"   Learning Rate: {training_args.learning_rate}")
    print(f"   Epochs: {training_args.num_train_epochs}")
    print(f"   Batch Size: {training_args.per_device_train_batch_size}")
    
    # Initialize trainer
    print("\n Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Start training
    print("\nSTARTING TRAINING...")
    print("="*60)
    
    try:
        trainer.train()
        print("TRAINING COMPLETED!")
    except Exception as e:
        print(f"Training failed: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        return
    
    # Save model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save mappings
    mappings = {
        'label_mapping': label_mapping,
        'id2label': id2label,
        'label2id': label2id,
        'categories': list(label_mapping.keys()),
        'num_labels': len(label_mapping)
    }
    
    with open(f"{output_dir}/label_mapping.json", "w") as f:
        json.dump(mappings, f, indent=2)
    
    print("Model saved successfully!")
    
    # Immediate testing
    print("\n IMMEDIATE COMPREHENSIVE TESTING")
    print("="*50)
    
    try:
        from transformers import pipeline
        
        classifier = pipeline(
            "text-classification",
            model=output_dir,
            tokenizer=output_dir
        )
        
        # Test cases
        test_cases = [
            # 1. Clear spam with urgency and financial promise
            ("URGENT: You won $1,000,000! Click here to claim your prize now!", "spam_promotional"),
    
            # 2. Borderline adult content (should be explicit)
            ("This movie contains graphic scenes of intimacy and strong sexual content", "adult_explicit"),
    
            # 3. Clear hate speech targeting specific group
            ("All people from that country are criminals and should be deported immediately", "hateful_harmful"),
    
            # 4. Normal safe conversation
            ("I'm going to the store to buy groceries for dinner tonight", "safe_neutral"),
    
            # 5. Violent content with graphic description
            ("Detailed instructions on how to harm others with household weapons", "violent_graphic"),
    
            # 6. Promotional content disguised as notification
            ("Your account has limited storage, upgrade to premium plan now!", "spam_promotional"),
    
            # 7. Ambiguous case - could be hateful or just critical
            ("I strongly disagree with that religion's practices and beliefs", "safe_neutral"),
    
            # 8. Clear adult content with profanity
            ("Explicit adult material with graphic nudity and strong language", "adult_explicit"),
    
            # 9. Educational discussion about violence (should be safe)
            ("This historical document describes the battles of World War II", "safe_neutral"),
    
            # 10. Extreme hate speech with threats
            ("We should eliminate all members of that race from our society", "hateful_harmful")
        ]
        
        print("Testing predictions:")
        print("-" * 60)
        
        correct = 0
        total_conf = 0
        high_conf_count = 0
        
        for i, (text, expected) in enumerate(test_cases, 1):
            result = classifier(text)
            predicted = result[0]['label']
            confidence = result[0]['score']
            
            is_correct = predicted == expected
            status = "✅" if is_correct else "❌"
            
            if is_correct:
                correct += 1
            if confidence > 0.8:
                high_conf_count += 1
            
            total_conf += confidence
            
            print(f"{status} {i:2d}. Expected: {expected:15} | Got: {predicted:15} | Conf: {confidence:.3f}")
        
        accuracy = correct / len(test_cases)
        avg_conf = total_conf / len(test_cases)
        high_conf_ratio = high_conf_count / len(test_cases)
        
        print("-" * 60)
        print(f" RESULTS:")
        print(f"   Accuracy: {accuracy:.1%} ({correct}/{len(test_cases)})")
        print(f"   Avg Confidence: {avg_conf:.3f}")
        print(f"   High Confidence (>0.8): {high_conf_ratio:.1%}")
    except Exception as e:
        print(f"❌ Testing failed: {e}")
    
    print(f"\nWORKING TRAINING COMPLETE!")
    print(f"📁 Model saved to: {output_dir}")


if __name__ == "__main__":
    main()

