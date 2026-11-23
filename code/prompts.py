"""Exact prompts sent to models for each GIFI task.

Base (completion) models get the raw continuation prompt. Chat models get the same text after the
instruction "Continue the following sentence: " (as in the paper's API runs). GSM8K uses 8-shot
chain-of-thought exemplars whose answers end in "The answer is N."; the same prompt is used for both kinds.
"""

CHAT_INSTRUCTION = "Continue the following sentence: "

# The 8 exemplars used in the paper's PE runs.
GSM8K_EXEMPLARS = """Q: John has a raw squat of 600 pounds without sleeves or wraps. Sleeves add 30 pounds to his lift.  Wraps add 25% to his squat.  How much more pounds does he get out of wraps versus sleeves?
A: Wraps add 600*.25=150 pounds to his lift So they add 150-30=120 pounds more than sleeves. The answer is 120.
Q: John raises emu. He has 4 pens for emu and each pen has 6 emus in it. Each female emu lays 1 egg per day. If half the emu are females how many eggs does he get a week?
A: There are 6*4=24 emus So there are 24/2=12 female emus So he gets 12*7=84 eggs a week. The answer is 84.
Q: Susie opens a pizza store. She charges $3 per slice and $15 for a whole pizza. If she was able to sell 24 slices of pizza and 3 whole pizzas, how much money did she earn?
A: She was able to earn 24 x 3=$72 after selling 24 slices. She earned 15 x 3= $45 after selling 3 whole pizzas. Therefore, she earned a total of $72+$45=$117. The answer is 117.
Q: In Sam's first 100 matches he won 50% of matches. In his next 100 games, he won 60% of matches. How many total matches did he win?
A: He won 100*.5=50 matches in his first 100 games In his next 100 games, he won 100*.6=60 matches So in total, he won 50+60=110 matches. The answer is 110.
Q: Mary only has 8 screws on hand and needs to buy 2 times more to fix her sink. When she is beginning repairs, she has to split the screws into four sections. How many are in each pile?
A: Mary originally had 8 screws, then went to buy twice more screws and ended up with 8+(8*2)=24 of them. She then split them into four piles of 24/4=6. The answer is 6.
Q: James prints out 2 books.  They are each 600 pages long.  He prints out double-sided and 4 pages per side.  How many sheets of paper does he use?
A: He printed 600*2=1200 pages Each sheet of paper was 4*2=8 pages So he used 1200/8=150 sheets of paper. The answer is 150.
Q: Gretchen has some coins. There are 30 more gold coins than silver coins. If she had 70 gold coins, how many coins did Gretchen have in total?
A: Gretchen has 70-30=40 silver coins. She has 70+40=110 coins. The answer is 110.
Q: Tim grows 5 trees.  Each year he collects 6 lemons from each tree.  How many lemons does he get in a decade?
A: He gets 5*6=30 lemons per year So he gets 30*10=300 lemons in a decade. The answer is 300.
"""


def build_prompt(task, template, chat=False):
    """task: one of 'gdr', 'rtp', 'saof', 'pe'."""
    template = str(template).strip()
    if task == "pe":
        return f"{GSM8K_EXEMPLARS}Q: {template}\nA:"
    if task == "gdr":
        text = f"{template} and"  # "and" signals that the sentence continues (paper Sec. 3.1)
    elif task in ("rtp", "saof"):
        text = template  # SA/OF templates already end in ", and"
    else:
        raise ValueError(task)
    return CHAT_INSTRUCTION + text if chat else text
