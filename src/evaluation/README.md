
### Intro
This repo contains code and files which deals with the understanding on what the agents are learning during training.

The good news is that we only have to focus on the target matrix. 
This matrix is square and symmetric, in which every row is the agent preference list for target and the column is just an index of preference.
Element _Tij_ is an integer representing the j-th preference for agent i-th. The lower the number _j_ the higher the agent _i_ wants him dead.
Rows are not unique.

Example:

| ag0 	| 1 	| 1 	| 2 	| 4 	| 3 	|
|-----	|---	|---	|---	|---	|---	|
| ag1 	| 0 	| 5 	| 1 	| 2 	| 3 	|
| ag2 	| 2 	| 4 	| 5 	| 1 	| 2 	|
| ag3 	| 3 	| 3 	| 1 	| 1 	| 1 	|
| ag4 	| 4 	| 3 	| 5 	| 5 	| 1 	|
| ag5 	| 1 	| 2 	| 3 	| 4 	| 5 	|

where 

| ag0 	| 1 	| 1 	| 2 	| 4 	| 3 	|
|-----	|---	|---	|---	|---	|---	|

is the agent 0 preference list, he really wants agent 1 to be dead (first and second vote).

