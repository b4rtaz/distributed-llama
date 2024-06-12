#!/bin/bash

# This is a simple test of generating a sequence that fulfills the KV cache.
#
# Used model & tokenizer: https://huggingface.co/b4rtaz/llama-3-8b-distributed-llama
# Probably, this test will be working correctly only on MacBook Pro M1, due to differences in float multiplication on different CPUs.

cd "$(dirname "$0")"
cd ..

# Source: https://www.opensourceshakespeare.org/views/plays/play_view.php?WorkID=macbeth&Scope=entire
PROMPT="Duncan. What bloody man is that? He can report,
As seemeth by his plight, of the revolt
The newest state. 20

Malcolm. This is the sergeant
Who like a good and hardy soldier fought
'Gainst my captivity. Hail, brave friend!
Say to the king the knowledge of the broil
As thou didst leave it. 25

Sergeant. Doubtful it stood;
As two spent swimmers, that do cling together
And choke their art. The merciless Macdonwald—
Worthy to be a rebel, for to that
The multiplying villanies of nature 30
Do swarm upon him—from the western isles
Of kerns and gallowglasses is supplied;
And fortune, on his damned quarrel smiling,
Show'd like a rebel's whore: but all's too weak:
For brave Macbeth—well he deserves that name— 35
Disdaining fortune, with his brandish'd steel,
Which smoked with bloody execution,
Like valour's minion carved out his passage
Till he faced the slave;
Which ne'er shook hands, nor bade farewell to him, 40
Till he unseam'd him from the nave to the chaps,
And fix'd his head upon our battlements. 

Duncan. O valiant cousin! worthy gentleman! 

Sergeant. As whence the sun 'gins his reflection
Shipwrecking storms and direful thunders break, 45
So from that spring whence comfort seem'd to come
Discomfort swells. Mark, king of Scotland, mark:
No sooner justice had with valour arm'd
Compell'd these skipping kerns to trust their heels,
But the Norweyan lord surveying vantage, 50
With furbish'd arms and new supplies of men
Began a fresh assault. 

Duncan. Dismay'd not this
Our captains, Macbeth and Banquo? 

Sergeant. Yes; 55
As sparrows eagles, or the hare the lion.
If I say sooth, I must report they were
As cannons overcharged with double cracks, so they
Doubly redoubled strokes upon the foe:
Except they meant to bathe in reeking wounds, 60
Or memorise another Golgotha,
I cannot tell.
But I am faint, my gashes cry for help. 

Duncan. So well thy words become thee as thy wounds;
They smack of honour both. Go get him surgeons. 65
[Exit Sergeant, attended]
Who comes here?"

GENERATED="Malcolm. The worthy Thane of Ross.
Duncan. What a haste looks through a duel's wounds! 70
Some must be pac'd. 
[Exit Ross]
See this encounter is like to the poring
On of a beggar's story, told by one
That means to pluck upon the heart the strings
And draw the tears thriftily. 75
[Enter Lennox]
How goes the night, boy?

Lennox. The night is long that none should wake.

Duncan. You do not need to stare. The Moor
To know the man. 'Tis the Moors devices. 80
[Exit Lennox]
By the happy right of mine own hands,
Strike all that live in this poor thing of mine.
'Tis calld the Eyrie, and I am sick at heart.
As hellish-devils do the damned souls
O'their bad lives, thus ill-breveted, linger
O'er lamps and forks and other instruments
That prove the stages of the night. 90
Good sir, take note; I bid you farewell:
Come sleep, and cut short this nitty romance.
[He sleeps.]
If cravens, I bear them like the Minion of the moon,
With tiptoe foot he sneaks and starts to be a man. 95
And when he is found asleep, awake him with this armed' s address: 
That sleep which th'assassin hallowed,
Scotland, awake; your king is murder'd, sleep no more. 100
*Furbish'd. Weapons polished for battle. 
*Thriftily. Fastidiously, thoughtfully. 
*Eyrie. Fortress; the lair of birds of prey. 
*Minion. A braggart, a coward. 

1.5

Macbeth. So foul and fair a day I have not seen. 5
Ross. Good morning, noble Macbeth. I come from Inverness,
And find our throne void, the arm'd rest you; 10
My Lord of Cassil has resigned his life.
Macbeth. Whate'er you owe, in time repay, fair friends.
Note you the words; I pray you do.
Ross. I am your faithful servant, and will keep
My sworn reward upon your life; my lord.
Macbeth. You shall be well rewarded; stay the press, 20
And I'll not fail. How now, good fellow?
Servant. Sir, his schoolmaster. 25
Macbeth. Well, good, though, old.
Tell me, good fellow, how goes the night? 30
Servant. There's marrygold and fire in your veins, my lord.
Macbeth. He does commend you; the weight of this old night's embargoes 35
Did one hour's waste of time lay upon him.
I know when we are too safe, 'tis dangerous to be secure;
Therefore our fearful parts do brave the danger 40
Which knows it not. I see you are a gentleman.
And a laudable one too; I am most off obliged.
Servant. I should be sorry, my good lord, to have had the labour 45
To outlive this damned hour. 50
Macbeth. What's done cannot be undone. To bed, to bed, to bed.
Servant. Will it please you to lie still? 55
Macbeth. Lord, lord, my heart is in my mouth. All's true that ends well.
Servant. I thank you, fair, and leave you to the content. 60
Macbeth. You see, my lord, it smokes, and shows no cause
Why the drone dies. 65
Servant. Grief fills the room up of one vast stair,
And downs our vaults to the inconstant man above. 70
Macbeth. Go bid thy masters and thy mistress say, 75
I have power in earth to do so much.
There's comfort yet. They are assailable. Then say I,
Thus ye may answer.
Servant. He cannot be wronged; or being wronged, 80
I cannot help him. 85
Macbeth. You know but by this; as this, 90
The Jew foole is hang'd. 95
Servant. No more today, my lord. 100
Macbeth. He does shame to tell him he loves him, but not remove him 105
From his true place; no.
Servant. That's true, and now I remember the story 110
Of that sign in Leo four diurnal courses
Returning in a constant motion were within 115
A boare that had on Taurus' back tetracted; 120
Or neuer, or but once in modulated accidence. 125
Macbeth. Thou climd'st alone, ty'd to the stag's horn.
Servant. I was a bull, for this the goodly year. 130
Come, put me in my place.
Macbeth. Now go to sleep. 135
Servant. The west neuer sett before the equinox 140
Till now; and sunnes look'd not theyr frequencie 145
Upon our lappe till now, my lord. 150
Macbeth. This game of chance you term a gong.
Servant. A gong is a scotch word for an egg. 155
Macbeth. Peace, be still. 160
Servant. I coniecture I smell the blood of an Englishman. 165
Macbeth. The faith is murthered.
Servant. That murder'd in his sleep. 170
Macbeth. And sleeping murdered. 175
Servant. In the fair queen heere in his royal court. 180
Macbeth. So great a mercy that it may last eternally.
Servant. The earth hath bubbles as the water hath, 185
And these are of them. Whate'er we will do 190
To mend the trespasses of the comming time 195
Shall be the seedes of new mischefe, and shall beget 200
The formes of the extinctnese, which we are now. 205
Macbeth. We have scorch'd the snake, not kill'd it. 210
Servant. They hunt it in the morn. Good gally, good lord! 215
It weares a gilded snout. 220
Macbeth. It is the very painting of your fear. 225
Servant. This is the worst. 230
Macbeth. A fair quater of a mile is yet to go. 235
Servant. A mile and half. 240
Macbeth. I have run fifteen miles to-day.
Servant. A calender's date.
Macbeth. A bigger patch, a bigger patch. 245
Servant. Thirteen of more. 250
Macbeth. Wast thou with him? 255
Servant. No, nor he to night. 260
Macbeth. Thou seest the moon"

echo "Generating, it can take a while..."

OUTPUT=$(( ./dllama generate --seed 12345 --temperature 0.9 --topp 0.9 --prompt "$PROMPT" --weights-float-type q40 --buffer-float-type f32 --nthreads 2 --steps 2048 --model models/llama3_8b_q40/dllama_model_llama3_8b_q40.m --tokenizer models/llama3_8b_q40/dllama_tokenizer_llama3_8b_q40.t --workers 127.0.0.1:9999 127.0.0.1:9998 127.0.0.1:9997 ) 2>&1)

echo "$OUTPUT"

if [[ $OUTPUT == *"$GENERATED"* ]]; then
    echo "✅ Output is same"
else
    echo "❌ Output is different"
fi
