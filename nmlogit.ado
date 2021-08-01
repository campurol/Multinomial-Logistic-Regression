
program define nmlogit


# create ado file
# test ado file with dummy data
# add choice-chooser variables



program define afgk
version 14.2
syntax, ///
	BOOT(integer)	///
	OUTPUT(string) 	///
	WEIGHTS(string)	///
	SIDE(string)	///
	NAICSlevel(integer) ///
	CRIT(real)		///
	TOLTYPE(string) ///
	MODEL(string) ///
	PATH(string)	///
	DB(string) ///
	[BOOTSTRAP(string) REPS(integer 0) HORSE MODEL2(string) WEIGHTS2(string) INIT(real -10) INIT2(real -10)]

	if "`horse'"!="" & "`model'"=="`model2'" & "`weights'"=="`weights2'"{
		display as error "you have to specify either different models or different weights in horse race model"
