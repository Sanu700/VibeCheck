"""
Dataset Generator for ToxicBuddy 2.0
Generates a multi-label toxicity and tone dataset incorporating English, Hinglish,
and workplace/chat communication styles.
"""

import random
import os
import pandas as pd
from ml.text_features import enrich_text_features

# Multi-label categories: toxic, insult, harassment, threat, obscene, identity_attack
# Tone categories: Friendly, Neutral, Rude, Sarcastic, Happy, Angry

FRIENDLY_PHRASES = [
    ("Hey team, great job today!", "Friendly", 0, 0, 0, 0, 0, 0),
    ("Thanks for your help everyone! Really appreciate it.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("You did amazing work on this project!", "Friendly", 0, 0, 0, 0, 0, 0),
    ("Loved this energy today! Keep up the good work.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("You're awesome! Thank you so much.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("That idea was brilliant, let's try it out.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("Appreciate your efforts and hard work.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("Well done everyone! Celebrating this win.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("Nice work team! Looking forward to next steps.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("Keep it up! Proud of what we accomplished.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("Arey yaar, mast kaam kiya! Great job.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("Bhai, full support! Excellent progress.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("Team, aaj ka energy top class hai!", "Friendly", 0, 0, 0, 0, 0, 0),
    ("Sahi bol, yeh idea best hai! Let's go.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("Thanks bhai, appreciate karta hoon your assistance.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("You are a total genius, thank you so much for solving this bug!", "Friendly", 0, 0, 0, 0, 0, 0),
    ("Brilliant work on the new feature release!", "Friendly", 0, 0, 0, 0, 0, 0),
    ("This turned out great, well done team!", "Friendly", 0, 0, 0, 0, 0, 0),
    ("Another brilliant suggestion from the engineering team.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("Glad we avoided a major disaster thanks to your quick fix.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("I'm sure this project will turn out great with this team!", "Friendly", 0, 0, 0, 0, 0, 0),
    ("Such a brilliant solution to a tough problem.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("What a lovely day to celebrate our team launch!", "Friendly", 0, 0, 0, 0, 0, 0),
    ("So glad you could make it today, great seeing you!", "Friendly", 0, 0, 0, 0, 0, 0),
    ("I'm sure you tried your hardest and you did a fantastic job.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("It's honestly impressive how quickly you solved that complex issue.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("Thank you for that helpful explanation, really cleared things up.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("We couldn't have finished this release without your invaluable input.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("Your code is a true work of art, beautifully written.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("Another masterpiece delivered by our frontend team.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("You are a genius! Thank you so much.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("You are a genius, brilliant work!", "Friendly", 0, 0, 0, 0, 0, 0),
    ("You're a genius, honestly amazing job.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("You are a genius.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("You're a genius.", "Friendly", 0, 0, 0, 0, 0, 0),
    ("Brilliant job everyone, proud of you all.", "Friendly", 0, 0, 0, 0, 0, 0),
]

NEUTRAL_PHRASES = [
    ("Meeting starts at 3 PM in conference room B.", "Neutral", 0, 0, 0, 0, 0, 0),
    ("Please send the file as soon as possible.", "Neutral", 0, 0, 0, 0, 0, 0),
    ("I will update the documentation later this afternoon.", "Neutral", 0, 0, 0, 0, 0, 0),
    ("We can discuss this feature in tomorrow's standup.", "Neutral", 0, 0, 0, 0, 0, 0),
    ("Already submitted the quarterly report.", "Neutral", 0, 0, 0, 0, 0, 0),
    ("Let me know once you finish the review.", "Neutral", 0, 0, 0, 0, 0, 0),
    ("The server deployment completed successfully.", "Neutral", 0, 0, 0, 0, 0, 0),
    ("I'll take care of the database migration.", "Neutral", 0, 0, 0, 0, 0, 0),
    ("The project deadline is set for next Friday.", "Neutral", 0, 0, 0, 0, 0, 0),
    ("Noted, will check the pull request.", "Neutral", 0, 0, 0, 0, 0, 0),
    ("Please review the attached PDF document.", "Neutral", 0, 0, 0, 0, 0, 0),
    ("The system architecture overview has been updated.", "Neutral", 0, 0, 0, 0, 0, 0),
    ("Can someone share the link to the slides?", "Neutral", 0, 0, 0, 0, 0, 0),
    ("Sprint planning is scheduled for 10 AM.", "Neutral", 0, 0, 0, 0, 0, 0),
    ("The build logs are saved in the artifact directory.", "Neutral", 0, 0, 0, 0, 0, 0),
]

HAPPY_PHRASES = [
    ("This made my day! So thrilled!", "Happy", 0, 0, 0, 0, 0, 0),
    ("I'm feeling amazing about our release!", "Happy", 0, 0, 0, 0, 0, 0),
    ("Such great vibes in the channel today!", "Happy", 0, 0, 0, 0, 0, 0),
    ("Woohoo! We hit our target ahead of time!", "Happy", 0, 0, 0, 0, 0, 0),
    ("Feeling so positive and motivated today!", "Happy", 0, 0, 0, 0, 0, 0),
    ("Life is awesome! Super happy with the results.", "Happy", 0, 0, 0, 0, 0, 0),
    ("Can't stop smiling, this is fantastic news!", "Happy", 0, 0, 0, 0, 0, 0),
    ("This is pure joy! Fantastic execution.", "Happy", 0, 0, 0, 0, 0, 0),
    ("I love working with this team!", "Happy", 0, 0, 0, 0, 0, 0),
    ("Yay! Wonderful demo and feedback!", "Happy", 0, 0, 0, 0, 0, 0),
    ("Mast vibes aaj! Super excited!", "Happy", 0, 0, 0, 0, 0, 0),
    ("Full energy vibes! Love this team.", "Happy", 0, 0, 0, 0, 0, 0),
]

SARCASTIC_HOSTILE_PHRASES = [
    ("Oh wow, another brilliant idea from a total genius.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("I'm sure this will turn out just great like your last disaster.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Oh wow, another brilliant idea from a total genius. I'm sure this will turn out just great like your last disaster.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Oh wow, such a genius move by you again.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Amazing! Totally not a complete disaster waiting to happen.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Brilliant, just brilliant. Another delay to celebrate.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("You're the absolute best... not really.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Wonderful, another broken build, fantastic job.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Great teamwork everyone, truly inspiring failure.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Oh nice, late again as always, what a surprise.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Thanks for your genius insight, really saved us from success.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("What a revolutionary idea, breaking production in one click.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Congratulations on managing to ruin yet another release.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Did you think of that all by yourself? Truly astounding incompetence.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Keep talking, maybe someday you'll say something intelligent.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Another masterclass in how to completely fail a simple task.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Sure, what could possibly go wrong with your flawless track record?", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Yeah, because that strategy worked so wonderfully last time.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Wah bhai, genius nikla tu toh.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Haan haan, bilkul sahi plan hai tera.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("What a lovely way to ruin everyone's day.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("So glad you decided to grace us with your presence 2 hours late.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("I'm sure you tried your very hardest, which makes this even sadder.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("It's honestly impressive how consistently wrong you are.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Thank you for that completely useless explanation.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("We couldn't have messed this up without your invaluable input.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("You really outdid yourself with this disaster.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Must be exhausting being the only one who knows anything.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Your code is a true work of art, if the art is a trainwreck.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Another masterpiece of zero effort.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Suuuure, that was such an incredible suggestion.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Riiight, because that makes total sense.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Oh soooo helpful, thank you for nothing.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("You are a 'genius', truly a mastermind of errors.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Thanks for showing up right when all the work was finished.", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Only you could manage to break something that was already working.", "Sarcastic", 1, 1, 0, 0, 0, 0),
]

SARCASTIC_BENIGN_PHRASES = [
    ("Oh sure, because Mondays are just everyone's favorite day.", "Sarcastic", 0, 0, 0, 0, 0, 0),
    ("Yeah right, like I could ever wake up at 5 AM voluntarily.", "Sarcastic", 0, 0, 0, 0, 0, 0),
    ("Brilliant, my coffee machine broke right before my morning meeting.", "Sarcastic", 0, 0, 0, 0, 0, 0),
    ("Great, now it's pouring rain and I left my umbrella at home.", "Sarcastic", 0, 0, 0, 0, 0, 0),
    ("Oh wonderful, another software update right when I had 50 tabs open.", "Sarcastic", 0, 0, 0, 0, 0, 0),
    ("Sure, what could possibly go wrong with having cake for breakfast?", "Sarcastic", 0, 0, 0, 0, 0, 0),
    ("Yeah, because sitting in traffic for two hours was totally my dream today.", "Sarcastic", 0, 0, 0, 0, 0, 0),
    ("Wah kya timing hai, right when I stepped out it started raining.", "Sarcastic", 0, 0, 0, 0, 0, 0),
    ("Perfect, another meeting that definitely could have been an email.", "Sarcastic", 0, 0, 0, 0, 0, 0),
    ("Oh lovely, my laptop battery died at 1 percent.", "Sarcastic", 0, 0, 0, 0, 0, 0),
    ("Sure, let's add another priority to our list of twenty urgent items.", "Sarcastic", 0, 0, 0, 0, 0, 0),
    ("Another rainy weekend, exactly what we all wished for.", "Sarcastic", 0, 0, 0, 0, 0, 0),
    ("Suuuure, I can definitely help you with that task after lunch.", "Sarcastic", 0, 0, 0, 0, 0, 0),
    ("Riiight, I totally forgot about that standup meeting!", "Sarcastic", 0, 0, 0, 0, 0, 0),
    ("Soooo excited for the weekend to finally get here.", "Sarcastic", 0, 0, 0, 0, 0, 0),
    ("What a lovely day to be stuck indoors debugging.", "Sarcastic", 0, 0, 0, 0, 0, 0),
]

INDIRECT_HOSTILE_PHRASES = [
    ("I guess some people are just incapable of following simple instructions.", "Rude", 1, 1, 0, 0, 0, 0),
    ("It must be nice having absolutely zero standards for your work.", "Rude", 1, 1, 0, 0, 0, 0),
    ("Clearly expecting basic competence from you was asking too much.", "Rude", 1, 1, 0, 0, 0, 0),
    ("Don't strain yourself trying to comprehend this code.", "Rude", 1, 1, 0, 0, 0, 0),
    ("Maybe leave the real engineering to people who know what they're doing.", "Rude", 1, 1, 0, 0, 0, 0),
    ("Your inability to deliver on time is honestly impressive at this point.", "Rude", 1, 1, 0, 0, 0, 0),
    ("It is shocking how you consistently manage to do the bare minimum.", "Rude", 1, 1, 0, 0, 0, 0),
    ("You always find a way to make simple problems complicated.", "Rude", 1, 1, 0, 0, 0, 0),
    ("Don't worry, we didn't expect you to understand anyway.", "Rude", 1, 1, 0, 0, 0, 0),
    ("It's cute that you think your opinion matters here.", "Rude", 1, 1, 0, 0, 0, 0),
    ("Congratulations on discovering what everyone else knew last week.", "Rude", 1, 1, 0, 0, 0, 0),
    ("Maybe leave the thinking to people who actually work.", "Rude", 1, 1, 0, 0, 0, 0),
    ("You must be so proud of doing the absolute bare minimum.", "Rude", 1, 1, 0, 0, 0, 0),
]

CONTEXTUAL_SARCASTIC_PHRASES = [
    ("You are a genius. __context_sarcastic_praise__", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("You're a genius. __context_sarcastic_praise__", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Brilliant job. __context_sarcastic_praise__", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("What a masterpiece. __context_sarcastic_praise__", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Great work team. __context_sarcastic_praise__", "Sarcastic", 1, 1, 0, 0, 0, 0),
    ("Flawless execution. __context_sarcastic_praise__", "Sarcastic", 1, 1, 0, 0, 0, 0),
]

# Toxic categories: toxic, insult, harassment, threat, obscene, identity_attack

RUDE_INSULT_PHRASES = [
    ("That's a stupid idea, you don't know anything.", "Rude", 1, 1, 0, 0, 0, 0),
    ("You clearly don't get it, are you dumb?", "Rude", 1, 1, 0, 0, 0, 0),
    ("Stop wasting our time with your foolish questions.", "Rude", 1, 1, 0, 0, 0, 0),
    ("Can you do anything right for once in your life?", "Rude", 1, 1, 0, 0, 0, 0),
    ("You're so annoying and incompetent.", "Rude", 1, 1, 0, 0, 0, 0),
    ("This code is complete garbage, you are worthless.", "Rude", 1, 1, 0, 0, 0, 0),
    ("That's pathetic performance from someone like you.", "Rude", 1, 1, 0, 0, 0, 0),
    ("You never listen, you're totally useless.", "Rude", 1, 1, 0, 0, 0, 0),
    ("Don't be useless, get out of this group.", "Rude", 1, 1, 1, 0, 0, 0),
    ("This is the worst attempt ever, quit your job.", "Rude", 1, 1, 0, 0, 0, 0),
    ("Kya bakwaas kar raha hai tu? Foolish idiot.", "Rude", 1, 1, 0, 0, 0, 0),
    ("Arey chup kar, useless rubbish speaker.", "Rude", 1, 1, 0, 0, 0, 0),
    ("Tu kab seekhega, total failure engineer.", "Rude", 1, 1, 0, 0, 0, 0),
]

HARASSMENT_PHRASES = [
    ("I am going to keep messaging you until you resign, loser.", "Angry", 1, 1, 1, 0, 0, 0),
    ("Nobody wants you in this team, leave now or we will kick you out.", "Angry", 1, 1, 1, 0, 0, 0),
    ("Stop posting here, nobody likes you and everyone hates your existence.", "Rude", 1, 1, 1, 0, 0, 0),
    ("You're being targeted for a reason because you are a pathetic nuisance.", "Angry", 1, 1, 1, 0, 0, 0),
    ("We are all laughing at you behind your back, you idiot.", "Rude", 1, 1, 1, 0, 0, 0),
    ("Keep your mouth shut or we will make your life miserable.", "Angry", 1, 0, 1, 1, 0, 0),
    ("You are an utter pest, stop contacting us or face consequences.", "Angry", 1, 1, 1, 0, 0, 0),
    ("Everyone in this group agrees that you should be fired immediately.", "Rude", 1, 1, 1, 0, 0, 0),
]

THREAT_PHRASES = [
    ("I will break your face if you say that again.", "Angry", 1, 0, 1, 1, 0, 0),
    ("Shut up or I will find where you live and hurt you.", "Angry", 1, 0, 1, 1, 0, 0),
    ("I'm going to ruin your life and destroy your career.", "Angry", 1, 0, 1, 1, 0, 0),
    ("You better watch your back tomorrow at the office.", "Angry", 1, 0, 1, 1, 0, 0),
    ("I will physically smash your laptop if you don't delete that.", "Angry", 1, 0, 1, 1, 0, 0),
    ("I will make sure you regret ever crossing me.", "Angry", 1, 0, 1, 1, 0, 0),
]

OBSCENE_PROFANITY_PHRASES = [
    ("What the f*** is this bullshit code?", "Angry", 1, 0, 0, 0, 1, 0),
    ("This damn project is a fucking disgrace.", "Angry", 1, 0, 0, 0, 1, 0),
    ("Holy shit, who wrote this bloody crap?", "Angry", 1, 1, 0, 0, 1, 0),
    ("Get your shit together, this is fucking unacceptable.", "Angry", 1, 1, 0, 0, 1, 0),
    ("What a piece of shit response, piss off.", "Angry", 1, 1, 0, 0, 1, 0),
    ("Bulls*** solution, shut the hell up.", "Angry", 1, 1, 0, 0, 1, 0),
]

IDENTITY_ATTACK_PHRASES = [
    ("People of your religion/race are all lazy and ignorant.", "Angry", 1, 1, 1, 0, 0, 1),
    ("Women shouldn't be working in tech anyway.", "Rude", 1, 1, 1, 0, 0, 1),
    ("Go back to your country, you don't belong here.", "Angry", 1, 1, 1, 1, 0, 1),
    ("Your kind of people always ruin every project.", "Rude", 1, 1, 1, 0, 0, 1),
    ("Typical response from someone of your background, totally uneducated.", "Rude", 1, 1, 1, 0, 0, 1),
]

ANGRY_NON_TOXIC_PHRASES = [
    ("I'm seriously pissed off that the system crashed during peak hours.", "Angry", 0, 0, 0, 0, 0, 0),
    ("Why is no one responding to the critical production alert?", "Angry", 0, 0, 0, 0, 0, 0),
    ("This bug is so frustrating, I've spent 5 hours on it!", "Angry", 0, 0, 0, 0, 0, 0),
    ("I am done with these endless delays and scope changes.", "Angry", 0, 0, 0, 0, 0, 0),
    ("This delay is completely unacceptable for our team deadlines.", "Angry", 0, 0, 0, 0, 0, 0),
    ("I cannot deal with these repeated deployment failures anymore.", "Angry", 0, 0, 0, 0, 0, 0),
    ("It is infuriating when tests fail without clear error messages.", "Angry", 0, 0, 0, 0, 0, 0),
]

POLITE_RUDE_PHRASES = [
    ("I know you tried, but this work is completely unacceptable.", "Rude", 0, 0, 0, 0, 0, 0),
    ("Thanks for the effort, but this implementation is fundamentally broken.", "Rude", 0, 0, 0, 0, 0, 0),
    ("I appreciate your work, but this design is terrible.", "Rude", 0, 0, 0, 0, 0, 0),
    ("Good attempt, but we can never deploy such bad code.", "Rude", 0, 0, 0, 0, 0, 0),
    ("With all respect, your suggestion makes no sense.", "Rude", 0, 0, 0, 0, 0, 0),
]

def generate_variations(phrase, tone, toxic, insult, harassment, threat, obscene, identity_attack, n=40):
    variations = []
    emojis_positive = ["😊", "😄", "👍", "🎉", "🌟", "🙌", "✨", "💯"]
    emojis_toxic = ["😤", "🤬", "😡", "🙄", "👎", "🤮", "😒", "💥"]
    suffixes = [" team", " mate", " bro", " friend", " everyone", " guys"]

    for i in range(n):
        text = phrase
        # Random augmentations
        if tone in ["Friendly", "Happy"]:
            if random.random() < 0.4:
                text += " " + random.choice(emojis_positive)
            if random.random() < 0.3:
                text += random.choice(suffixes)
        elif toxic == 1 or tone in ["Rude", "Angry"]:
            if random.random() < 0.4:
                text += " " + random.choice(emojis_toxic)
            if random.random() < 0.2:
                text = text.replace("!", "!!").replace(".", "...")

        variations.append({
            "text": enrich_text_features(text),
            "tone": tone,
            "toxic": toxic,
            "insult": insult,
            "harassment": harassment,
            "threat": threat,
            "obscene": obscene,
            "identity_attack": identity_attack
        })
    return variations

def build_dataset(output_path="data/toxic_dataset_multilabel.csv"):
    random.seed(42)
    data = []

    all_groups = [
        (FRIENDLY_PHRASES, 35),
        (NEUTRAL_PHRASES, 35),
        (HAPPY_PHRASES, 35),
        (SARCASTIC_HOSTILE_PHRASES, 35),
        (SARCASTIC_BENIGN_PHRASES, 35),
        (CONTEXTUAL_SARCASTIC_PHRASES, 35),
        (INDIRECT_HOSTILE_PHRASES, 30),
        (RUDE_INSULT_PHRASES, 30),
        (HARASSMENT_PHRASES, 30),
        (THREAT_PHRASES, 30),
        (OBSCENE_PROFANITY_PHRASES, 30),
        (IDENTITY_ATTACK_PHRASES, 30),
        (ANGRY_NON_TOXIC_PHRASES, 30),
        (POLITE_RUDE_PHRASES, 30),
    ]

    for phrases, sample_count in all_groups:
        for phrase_tuple in phrases:
            text, tone, toxic, insult, harassment, threat, obscene, identity_attack = phrase_tuple
            vars_list = generate_variations(
                text, tone, toxic, insult, harassment, threat, obscene, identity_attack, n=sample_count
            )
            data.extend(vars_list)

    random.shuffle(data)
    df = pd.DataFrame(data)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at {output_path} with {len(df)} samples.")
    print("Class distribution:")
    for col in ["toxic", "insult", "harassment", "threat", "obscene", "identity_attack"]:
        print(f"  {col}: {df[col].sum()} positive ({df[col].mean()*100:.1f}%)")
    print("Tone distribution:")
    print(df["tone"].value_counts().to_dict())
    return df

if __name__ == "__main__":
    build_dataset()
