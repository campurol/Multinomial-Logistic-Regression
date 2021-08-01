import delimited "/Users/rcampusanog/Dropbox/Respaldo/Academico/UofT/Ideas/location/code/nmlogit/data/train_SINBKK_RT_B.csv", clear
keep if choice==1
expand 2, generate(expanded)
replace session_id = session_id * (expanded + 1)

expand 2, generate(expanded2)
replace session_id = session_id * (expanded2 + 2)

replace session_id = session_id * 2 if expanded==1
generate group = 1 if expanded ==0 & expanded2==0
replace group = 2 if expanded ==1 & expanded2==0
replace group = 3 if expanded ==1 & expanded2==1
replace group = 4 if group==.


replace alter_id = floor(60*runiform()) if group==1 | group==4
replace alter_id = floor(120*runiform()) if group==2
replace alter_id = floor(10*runiform()) if group==3
gen group2= "hola"

preserve
	collapse (mean) reco_contains_mh reco_contains_tg reco_contains_pg reco_contains_sq reco_contains_vn reco_contains_cx reco_contains_od, by(group group2 alter_id)
	export delimited using "/Users/rcampusanog/Dropbox/Respaldo/Academico/UofT/Ideas/location/code/nmlogit/data/train_SINBKK_RT_B_alter_id.csv", replace
restore

preserve
	collapse (mean) deptime_outbound_sin2p deptime_outbound_sin4p deptime_outbound_cos2p deptime_outbound_cos4p deptime_inbound_sin2p deptime_inbound_sin4p deptime_inbound_cos2p deptime_inbound_cos4p, by(session_id)
	export delimited using "/Users/rcampusanog/Dropbox/Respaldo/Academico/UofT/Ideas/location/code/nmlogit/data/train_SINBKK_RT_B_session_id.csv", replace
restore
keep session_id alter_id group  choice group2


export delimited using "/Users/rcampusanog/Dropbox/Respaldo/Academico/UofT/Ideas/location/code/nmlogit/data/train_SINBKK_RT_B_choices.csv", replace


