## Metrics

|       | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE_L | CIDEr |
| ----- | ------ | ------ | ------ | ------ | ------ | ------- | ----- |
| human |        |        | 0.290  | 0.192  | 0.240  | 0.465   | 0.849 |
| G-MLE |        |        | **0.393**  | **0.299**  | **0.248**  | **0.527**   | **1.020** |
| G-GAN |        |        | 0.305  | 0.207  | 0.224  | 0.475   | 0.795 |
| ours  | 0.381  | 0.224  | 0.139  | 0.090  | 0.153  | 0.393   | 0.680 |



## Image Captioning Results

### Positive

|      | ![](zebra_giraffe.jpg)                            | ![](man_cook.jpg)                               | ![](sink.jpg)                                  |
| ---- | ------------------------------------------------- | ----------------------------------------------- | ---------------------------------------------- |
| z1   | a zebra is standing in the green grass .          | a man is standing next to a stove .             | a bathroom with a sink , mirror and a mirror . |
| z2   | two zebras are grazing in the grass field .       | a man standing in a kitchen preparing food .    | a bathroom sink is next to a mirror            |
| z3   | a giraffe standing next to a giraffe in a field . | a person standing in front of kitchen counter . | a bathroom with a sink and a mirror            |









### Negative

|      |   ![](train.jpg)   |   ![](cellphone.jpg)   |  ![](train2.jpg)   |
| ---- | ---- | ---- | ---- |
| z1 | a man is parked on the train station . | a person is taking a picture of a cigarette . | a man is parked on the train tracks . |
| z2 | a man is traveling down a train station . | a person is holding a cell phone in his hand . | a red train is sitting on the tracks . |
| z3 | a man is parked at a train station . | a man is holding a cell phone . | a silver and white and a train tracks . |



## Captions Property

### Predicted captions

```
a plate with a slice of a table .
a close up of a pizza on a pizza .
a pizza with a table with a plate .
a pizza on a table with a pizza .
a plate with a slice of a plate .
a pizza with a table with a plate .
a man with a table with a table .
a man holding a table with a table .
a man sitting at a skateboard .
a man on a white and a skateboard .
a man in a white and white .
a giraffe standing next to a tree .
a giraffe is standing in a field .
a zebra and a zebra in a field .
a couple of zebras in a zebra .
a zebra and a man in a field .
a zebra standing in a field .
a zebra standing on a grass covered field .
a zebra standing on a lush green field .
a zebra standing on a lush green field .
a white and a white frisbee .
a large a white and white frisbee .
a close up a white and white frisbee .
a white and a white frisbee .
a man holding a white frisbee .
a person is holding a white frisbee .
a man on a skateboard on a street .
a stop sign with a street .
a stop sign on a street sign .
a small airplane is on a plane .
a plane sitting on a runway .
a large airplane on a runway .
a man on a skateboard on a street .
a white and a white frisbee .
a man holding a white frisbee .
a person is holding a white frisbee .
a man on a skateboard on a street .
a stop sign on a street sign .
a small airplane is on a plane .
a small airplane is on a runway .
a large airplane on a runway .
a man on a skateboard on a street .
a green bananas on a banana .
a banana sitting on a white plate .
a close up with a banana .
a man sitting on a cell phone .
a person is holding a cell phone .
a person is holding a cell .
a white toilet sitting next to a sink .
a man sitting on a train .
a man in a red and white train .
a man standing on a train .
a man on a skateboard on a street .
a man sitting on a skateboard .
a man sitting on a bench .
a person is sitting on a white frisbee .
a person is sitting on a red .
a man is holding a tennis racquet .
a cat sitting on a table .
```



real captions

```
a bathroom with a border of butterflies and blue paint on the walls above it .
veral metal balls sit in the sand near a group of people .
a kitchen with brown cabinets , tile backsplash , and grey counters .
a panoramic view of a kitchen and all of its appliances .
a blue and white bathroom with butterfly themed wall tiles .
a <unk> stop sign across the street from a red car
a vandalized stop sign and a red beetle on the road
the vanity contains two sinks with a towel for each .
a sink and a toilet inside a small bathroom .
a white square kitchen with tile floor that needs repair
a panoramic photo of a kitchen and dining room
an angled view of a beautifully decorated bathroom .
the two people are walking down the beach .
an empty kitchen with white and black appliances .
two people carrying surf boards on a beach .
a very clean and well decorated empty bathroom
two women preparing food in a kitchen , one at the sink and one at the tabl
a person , protected from the rain by their umbrella , walks down the road .
a surfer , a woman , and a child walk on the beach .
a cat stuck in a car with a slightly opened window .
white pedestal sink and toilet located in a poorly lit bathroom .
a toilet , sink , bathtub with shower , mirror and cabinet
a white kitchen in a home with the light on .
two bicycles and a woman walking in front of a shop
a brown horse is grazing grass near a red house .
a kitchen with a countertop that includes an apple phone .
a few people sit on a dim transportation system .
the bathroom with a toilet has an interesting sink .
a bicycle is parked by a bench at night .
a black car is near someone riding a bike .
this is a pick-up game of shirts and <unk> basketball
```





