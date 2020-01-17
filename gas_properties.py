def CH4(mw_atmos, mh = 1):
	"""Defines properties for CH4 as condensible"""
	if mh != 1: print("Warning: no M/H Dependence in CH4")
	gas_mw = 16.0
	# Lodders(2003)
	gas_mmr = 4.9e-4 * (gas_mw/mw_atmos) 
	#V.G. Manzhelii and A.M. Tolkachev, Sov. Phys. Solid State 5, 2506 (1964)
	rho_p =  0.49   #solid (T = 213 K)
	return gas_mw, gas_mmr, rho_p

def NH3(mw_atmos, mh = 1):
	"""Defines properties for  NH3 as condensible"""
	if mh != 1: print("Warning: no M/H Dependence in NH3")
	gas_mw = 17.
	#Lodders(2003)
	gas_mmr = 1.34e-4 * (gas_mw/mw_atmos) *1.0
	#V.G. Manzhelii and A.M. Tolkachev, Sov. Phys. Solid State 5, 2506 (1964)
	rho_p =  0.84  #solid (T = 213 K)
	return gas_mw, gas_mmr, rho_p

def H2O(mw_atmos, mh = 1):
	"""Defines properties for H2O as condensible"""
	if mh != 1: print("Warning: no M/H Dependence in H2O")
	gas_mw = 18.
	#Lodders(2003)
	gas_mmr = 7.54e-4 * (gas_mw/mw_atmos)  *1.0
	#V.G. Manzhelii and A.M. Tolkachev, Sov. Phys. Solid State 5, 2506 (1964)
	rho_p =  0.93   #solid (T = 213 K)
	return gas_mw, gas_mmr, rho_p

def Fe(mw_atmos, mh = 1):
	"""Defines properties for Fe as condensible"""
	if mh != 1: print("Warning: no M/H Dependence in Fe")
	gas_mw = 55.845
	gas_mmr = 1.30e-3
	#Lodders and Fegley (1998)
	rho_p =  7.875	
	return gas_mw, gas_mmr, rho_p

def KCl(mw_atmos, mh = 1):
	"""Defines properties for KCl as condensible"""	
	gas_mw = 74.5

	if mh ==1: 
		#SOLAR METALLICITY (abunds tables, 900K, 1 bar)
		gas_mmr = 2.2627E-07 * (gas_mw/mw_atmos)
	elif mh == 10:
		#10x SOLAR METALLICITY (abunds tables, 900K, 1 bar)
		gas_mmr = 2.1829E-06 * (gas_mw/mw_atmos)
	elif mh==50:
		#50x SOLAR METALLICITY (abunds tables, 900K, 1 bar)
		gas_mmr = 8.1164E-06 * (gas_mw/mw_atmos)
	else: 
		raise Exception("KCl gas mmr can only be computed for 1, 10 and 50x Solar Meallicity")
	#source unknown
	rho_p =  1.99
	return gas_mw, gas_mmr, rho_p

def MgSiO3(mw_atmos, mh = 1):
	"""Defines properties for MgSiO3 as condensible"""	
	if mh != 1: print("Warning: no M/H Dependence in MgSiO3")
	gas_mw = 100.4
	gas_mmr = 2.75e-3
	#Lodders and Fegley (1998)
	rho_p =  3.192
	return gas_mw, gas_mmr, rho_p

def Mg2SiO4(mw_atmos, mh = 1):
	"""Defines properties for Mg2SiO4 as condensible"""	
	if mh != 1: print("Warning: no M/H Dependence in MgSiO4")
	gas_mw = 140.7
	#NEW FORSTERITE (from Lodders et al. table, 1000mbar, 1900K)
	gas_mmr = 7.1625e-05/2 * (gas_mw/mw_atmos)  
	#Lodders and Fegley (1998)
	rho_p =  3.214	
	return gas_mw, gas_mmr, rho_p

def MnS(mw_atmos,mh=1):	
	"""Defines properties for MnS as condensible"""	

	gas_mw = 87.00

	gas_mmr =  mh * 6.37e-7 * (gas_mw/mw_atmos) 

	#Lodders and Fegley (2003) (cvm)
	rho_p =  4.0
	return gas_mw, gas_mmr, rho_p

def ZnS(mw_atmos, mh=1):
	"""Defines properties for ZnS as condensible"""	

	gas_mw = 97.46

	gas_mmr = mh*8.40e-8 * (gas_mw/mw_atmos) 

	#Lodders and Fegley (2003) (cvm)
	rho_p =  4.04	
	return gas_mw, gas_mmr, rho_p

def Cr(mw_atmos, mh=1):
	"""Defines properties for Cr as condensible"""	
	gas_mw = 51.996
	if mh==1:
		gas_mmr = 8.80e-7 * (gas_mw/mw_atmos) 
	elif mh==10:
		gas_mmr = 8.6803E-06 * (gas_mw/mw_atmos) 
	elif mh==50:
		gas_mmr = 4.1308E-05 * (gas_mw/mw_atmos) 
	else: 
		raise Exception("Chromium (Cr) Gas mmr not available for metallicity="+str(mh))
	
	#Lodders and Fegley (2003) (cvm)
	rho_p =  7.15
	return gas_mw, gas_mmr, rho_p

	




