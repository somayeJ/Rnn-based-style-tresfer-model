attention 4  , based on 3.3 with removing label part of the z, and zi sand concat target label, with the linear layer of the label being concatenated in the decoder scope


	
		
Version3 is different from version1&2:
	1. the FF layers for quary and values also have no  bias
	2. passing h(hidden state) to attention function
	3. remove scopes for W1, W2 & W3

Version2 is different from version1, in the following senses:
	1. the third FF layer has no bias
	2. the initializations is with the normal initializers

*****************************************************************
