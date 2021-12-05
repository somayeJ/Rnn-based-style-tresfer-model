
Version3.2.2 is based on version3.2.1(a, rho =1), but:
		we remove label part of the z, and zi at each step of generation, and concat them with the densed target label (which is passed through a linear layer )

to run this model with gyafc_preprocessed, i change 21, 20 tp 31 and 30 in all files:
adv_sds_attention.py, nn_td_v1.py, utils_v_1.py



create_z_SDS_attention3.2.2.py is the code to run to create the cnx vectors && attention weights



 
	
