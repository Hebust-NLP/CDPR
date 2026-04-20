benchmarks = {
    'MIntRec':{
        'intent_labels': [
                    'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize', 
                    'Agree', 'Taunt', 'Flaunt', 
                    'Joke', 'Oppose', 
                    'Comfort', 'Care', 'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave', 
                    'Prevent', 'Greet', 'Ask for help' 
        ],
        'binary_maps': {
                    'Complain': 'Emotion', 'Praise':'Emotion', 'Apologise': 'Emotion', 'Thank':'Emotion', 'Criticize': 'Emotion',
                    'Care': 'Emotion', 'Agree': 'Emotion', 'Taunt': 'Emotion', 'Flaunt': 'Emotion',
                    'Joke':'Emotion', 'Oppose': 'Emotion', 
                    'Inform':'Goal', 'Advise':'Goal', 'Arrange': 'Goal', 'Introduce': 'Goal', 'Leave':'Goal',
                    'Prevent':'Goal', 'Greet': 'Goal', 'Ask for help': 'Goal', 'Comfort': 'Goal'
        },
        'binary_intent_labels': ['Emotion', 'Goal'],
        'max_seq_lengths':{
            'text': 30, # truth: 26 
            'video': 230, # truth: 225
            'audio': 480, # truth: 477
        }
    },

    'MIntRec2.0': {
        'intent_labels': [
            'Acknowledge', 'Advise', 'Agree', 'Apologise', 'Arrange', 
            'Ask for help', 'Asking for opinions', 'Care', 'Comfort', 'Complain', 
            'Confirm', 'Criticize', 'Doubt', 'Emphasize', 'Explain', 
            'Flaunt', 'Greet', 'Inform', 'Introduce', 'Invite', 
            'Joke', 'Leave', 'Oppose', 'Plan', 'Praise', 
            'Prevent', 'Refuse', 'Taunt', 'Thank', 'Warn',
        ],
        'speaker_list' : ['friends person5', 'Cheyenne', 'Tate', 'Joey', 'Chandler', 'Dina', 'Myrtle', 'Sheldon', 'Garrett', 'Amy', 'Rajesh', 'Justine', 'friends person4', 
            'Leonard', 'friends person2', 'Janet', 'Jerry', 'Glenn', 'big bang person1', 'Penny', 'superstore person1', 'superstore person2', 'superstore person4',
            'Sandra', 'friends person1', 'Carol', 'Monica', 'Phoebe', 'Jeff', 'big bang person3', 'Emily', 'Gleen', 'Rachel', 'Howard', 'Adam', 'Ross', 'big bang person2', 
            'superstore person3', 'Nico', 'Kelly', 'Bo', 'big bang person4', 'Jonah', 'Bernadette', 'Cody', 'Marcus', 'friends person3'
        ],
        'max_seq_lengths': {
            'text': 50, 
            'video': 180,
            'audio': 400,
        }
    },

    'MELD-DA':{
        'intent_labels': ['a', 'ag', 'ans', 'ap', 'b', 'c', 'dag', 'g', 'o', 'oth', 'q', 's'],
        'label_maps': {
                    'a': 'Acknowledge', 'ag': 'Agreement', 'ans': 'Answer', 'ap': 'Apology', 
                    'b': 'Backchannel', 'c': 'Command', 'dag': 'Disagreement', 'g': 'Greeting',
                    'o': 'Statement Opinion', 'oth': 'Others', 'q': 'Question', 's': 'Statement Non Opinion'
        },
        'max_seq_lengths':{
            'text': 70, 
            'audio': 530, 
            'video': 250,
        }
    },

    'IEMOCAP-DA':{
        'intent_labels': ['a', 'ag', 'ans', 'ap', 'b', 'c', 'dag', 'g', 'o', 'oth', 'q', 's'],
        'label_maps': {
                    'a': 'Acknowledge', 'ag': 'Agreement', 'ans': 'Answer', 'ap': 'Apology', 
                    'b': 'Backchannel', 'c': 'Command', 'dag': 'Disagreement', 'g': 'Greeting',
                    'o': 'Statement Opinion', 'oth': 'Others', 'q': 'Question', 's': 'Statement Non Opinion'
        },
        'max_seq_lengths': {
            'text': 44,
            'video': 230,
            'audio': 380,
        }
    },
}