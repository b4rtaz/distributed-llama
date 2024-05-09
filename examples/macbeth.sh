#!/bin/bash

# This is a simple test of generating a sequence that fulfills the KV cache.
#
# Model: https://huggingface.co/b4rtaz/llama-3-8b-distributed-llama
# Probably, this test will be working correctly only on MacBook Pro, due to differences in float multiplication on different CPUs.

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
Duncan. What a haste looks through a troop? and when may
No sooner had this battle fought, than, ingrate and ungracious! 70
Who leap'd my back, and thence hasten'd me away,
And follows so he wins. 

Malcolm. As I do live, my lord, so happily prosper I, without this blow might have o'erpaid the world: 75
He loves our Majesty as boundlessly as we
Muster ourselves, and make a full battalion. 
Duncan. Then enter, sir, and alone with me great battles 80
I'll strain upon thy forehead, to this day
It is a faith, and makes the fire that burn in my veins: 
Thou hast it now, king, afield to-morrow. 
God be wi' you, father. 

Duncan. Farewell, farewell! or let me hear from you. 85
[Exeunt] 

THE THIRD SCENE
Macbeth, Banquo, Ross, and Angus.
Macbeth. So fair and foul a day I have not seen. 
It is calm, and yet is all together. 
[Thunder, then rain]
Look, how the blood of Sweden flows from hence;
The time is free. I see the Capitol; 90
The city of kingly eyes. 
[Thunder, then lightening]
And the remote parts of Parliament,
Which now behold, now can behold no more 95
In which time will appear how much
I have translated the flesh of Banquo
Into a crow that strays about the capital. 
[Treble knocks]
The Prince of Cumberland; that is a step
On which I must fall down, or else o'erleap, 100
For in my way it lies. Stars, hide your fires;
Let not light see my black and deep desires:
The eye wink at the hand; yet let that be
Which the eye fears to look upon.
Let it be a sin; 105
That the so lusterous, and so bright, so good
Should but be seen a fellow to my crime;
And dupe so ruin'd. 
[Treble knocks]
That my keen knife
See you satisfied. 
[Treble knocks] 110
Go to thy death. 
[Re-enter a Servant] 
How now, what names. 
Servant. Name. Marrow, marrow; that is the very question that I put to thee, 115
That is the very question that I put to thee,
Macbeth. Thou 'rt mad that thy sword is not temper'd. 
What you lack of temperance, that I lack in valour.
Art not without ambition; but without 120
The illness should attend it. What thou wouldst highly,
That wouldst thou boldly, and with thy virtues else
Wouldst thou have wildly holden; let fall thy hand;
[To the Servant]
Kilt him like a boar. 125
Macbeth. From this time these woes we will re-assume:
Not from our fingers' ends? We still have left
A special will to thrust these thorns more firmly.
A little more the wisely. Gently the weather. 130
[Treble knocks]
The Prince of Cumberland; that is a step
On which I must fall down, or else o'erleap,
For in my way it lies. Stars, hide your fires;
Let not light see my black and deep desires: 135
The eye wink at the hand; yet let that be
Which the eye fears to look upon.
Let it be a sin; that the so lusterous, and so bright, so good
Should but be seen a fellow to my crime; 140
And dupe so ruin'd. 
[Treble knocks] 
That my keen knife
See you satisfied. 
[Treble knocks] 
Go to thy death. 
[Re-enter a Servant] 
How now, what names. 
Servant. Name. Marrow, marrow; that is the very question that I put to thee, 145
That is the very question that I put to thee,
Macbeth. Thou 'rt mad that thy sword is not temper'd. 
What you lack of temperance, that I lack in valour.
Art not without ambition; but without 150
The illness should attend it. What thou wouldst highly,
That wouldst thou boldly, and with thy virtues else
Wouldst thou have wildly holden; let fall thy hand;
[To the Servant]
Kilt him like a boar. 155
Macbeth. From this time these woes we will re-assume:
Not from our fingers' ends? We still have left
A special will to thrust these thorns more firmly.
A little more the wisely. Gently the weather. 160
[Treble knocks]
Come, love, and we will a while chastise
That dares come to this.
[Re-enter a second Servant] 
What is that which caugh your eyes? 165
Second Servant. My young lord, I can tell. 
To think that they may see such sights!
And yet not be the eyes itself that see but, as 'tis said, a man should be the righter part of nature; if he be such, he need not
come behindhand too. 170
'Tis no time to cloak our faults. 
[Re-enter a third Servant] 
The very firstlings of my heart shall be
The firstlings of my head; I'll be their patriarch.
Come, put on gaiter; come, come, good mother, 175
Damned entrance of weather! 
[Thunder]
Come, get you to my woman's breasts; And on them give, and mercy onen me, let fall your holy disinclinations. 
[Exeunt]
Act III. SCENE 1.
The scene opens with the arrival of the King and his entourage at the castle of the thane of Fife. King Duncan, having heard of Macbeth's new successes, asks his thanes to rejoice with him. Macbeth's great respect for the king makes him slightly uncomfortable. King Duncan's concern for Macbeth's wife and children is further evidence of the king's warmth and loving nature. Macbeth appears ill at ease, perhaps at Duncan's evident concern. His language is overly formal and self-conscious, while his wife speaks rather bluntly. After Macbeth, the king, and his attendants enter, Banquo asks how Lady Macbeth is. 

Macbeth. When I am gone, 180
After life's fitful fever, he sleeps well, 
Though the powers of the strong world do set themselves 
Against his estate. 
King Duncan. So well to do! 
Had he his heart's desire, he 'd stoop 
To what humility 185
Might become the matter. 
Macbeth. As the matter now I 've put it. 
King Duncan. Well then, 190
Since that you are a father, show the child
The taking off, and that which now you do 195
Commit"

echo "Generating, it can take a while..."

OUTPUT=$(( ./main generate --seed 12345 --temperature 0.9 --topp 0.9 --prompt "$PROMPT" --weights-float-type q40 --buffer-float-type q80 --nthreads 8 --steps 2048 --model converter/dllama_meta-llama-3-8b_q40.bin --tokenizer converter/dllama_meta-llama3-tokenizer.t ) 2>&1)

echo "$OUTPUT"

if [[ $OUTPUT == *"$GENERATED"* ]]; then
    echo "✅ Output is same"
else
    echo "❌ Output is different"
fi
