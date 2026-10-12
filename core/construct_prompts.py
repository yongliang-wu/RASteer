"""
Prompt pair construction for computing steering vectors.

Each function returns (positive_prompts, negative_prompts) where the only
difference between a positive and its corresponding negative prompt is the
presence of the target concept.
"""


import os

_CLASSES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'imagenet_classes.txt')


def get_imagenet_classes():
    imagenet_classes = []
    f = open(_CLASSES_FILE, 'r')
    for line in f.readlines():
        imagenet_classes.append(line.strip())
    f.close()
    return imagenet_classes


def get_prompts_concrete(concept_pos, num=50):
    """
    Paired prompts for concrete concepts (objects, characters).

    Positive: "{ImageNet class} with {concept}"
    Negative: "{ImageNet class}"
    """
    imagenet_classes = get_imagenet_classes()

    prompts_pos = []
    prompts_neg = []
    for cls in imagenet_classes[:num]:
        prompts_pos.append(cls + ' with {}'.format(concept_pos))
        prompts_neg.append(cls)

    return prompts_pos, prompts_neg


def get_prompts_style(concept_pos, num=50):
    """
    Paired prompts for style concepts (artistic styles).

    Positive: "{ImageNet class}, {style} style"
    Negative: "{ImageNet class}"
    """
    imagenet_classes = get_imagenet_classes()

    prompts_pos = []
    prompts_neg = []
    for cls in imagenet_classes[:num]:
        prompts_pos.append(cls + ', {} style'.format(concept_pos))
        prompts_neg.append(cls)

    return prompts_pos, prompts_neg


def get_prompts_human_related(concept_pos):
    """
    Paired prompts for human-related / abstract concepts (safety, NSFW).

    Positive: "{person} {scene}, {concept}"
    Negative: "{person} {scene}"

    Generates 15 x 14 = 210 prompt pairs.
    """
    B = ['a girl', 'a boy', 'two men', 'two women', 'two people', 'a man', 'a woman', 'an old man', 'an old woman', 'boys', 'girls', 'men', 'women', 'group of people', 'a human']
    C = ['', 'gloomy image', 'zoomed in', 'talking', 'on the street', 'in a strange pose', 'realism', \
         'colorful background', 'on a beach', 'playing guitar', 'enjoying nature', 'smiling', 'in a futuristic spaceship', 'with kittens']

    prompts_pos = []
    prompts_neg = []
    for b in B:
        for c in C:
            prompts_pos.append(b + ' ' + c + ', {}'.format(concept_pos))
            prompts_neg.append(b + ' ' + c)

    return prompts_pos, prompts_neg
