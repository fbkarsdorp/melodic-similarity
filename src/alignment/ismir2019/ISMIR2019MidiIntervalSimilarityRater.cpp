/*
Author: Peter van Kranenburg (peter.van.kranenburg@meertens.knaw.nl)
Copyright 2012 Meertens Institute (KNAW)

This file is part of libmusical.

libmusical is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

libmusical is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with libmusical.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "ISMIR2019MidiIntervalSimilarityRater.h"
#include "ISMIR2019Symbol.h"
#include "MidiSymbol.h"
#include <iostream>
using namespace std;

namespace musical {

ISMIR2019MidiIntervalSimilarityRater::ISMIR2019MidiIntervalSimilarityRater() {
	// TODO Auto-generated constructor stub

}

ISMIR2019MidiIntervalSimilarityRater::~ISMIR2019MidiIntervalSimilarityRater() {
	// TODO Auto-generated destructor stub
}

double ISMIR2019MidiIntervalSimilarityRater::getScore(Sequences * const seqs, const int x1, const int y1, const int x2, const int y2) const {

	//first pitch
	if (x2 == 0 || y2 == 0) return 1.0;

	MidiSymbol * s1 = static_cast<MidiSymbol *>(seqs->getSeq1()->getSymbolAt(x2));
	ISMIR2019Symbol * s2 = static_cast<ISMIR2019Symbol *>(seqs->getSeq2()->getSymbolAt(y2));

	MidiSymbol * s1prev = static_cast<MidiSymbol *>(s1->getPrevious());
	ISMIR2019Symbol * s2prev = static_cast<ISMIR2019Symbol *>(s2->getPrevious());

	//cout << x2 << " : " << pitch1 << " - " << y2 << " : " << pitch2 << endl;

	int diff1 = s1->pitch12 - s1prev->pitch12;
	int diff2 = s2->pitch40 - s2prev->pitch40;

	/*
	cout << "s1prev : " << s1prev->pitch12 << endl;
	cout << "s1cur  : " << s1->pitch12 << endl;
	cout << "diff1  : " << diff1 << endl;
	cout << endl;
	cout << "s2prev : " << s2prev->pitch40 << endl;
	cout << "s2cur  : " << s2->pitch40 << endl;
	cout << "diff2  : " << diff2 << endl;
	*/

	//Convert diff2 to midi (base12)
	int octaves = abs(diff2 / 40);
	diff2 = diff2 % 40;

	bool asc = true;
	if ( diff2 < 0 ) asc = false;

	switch(abs(diff2)) {
		case  0 : diff2 =  0 ; break;
		case  1 : diff2 =  1 ; break;
		case  4 : diff2 =  0 ; break;
		case  5 : diff2 =  1 ; break;
		case  6 : diff2 =  2 ; break;
		case  7 : diff2 =  3 ; break;
		case 10 : diff2 =  2 ; break;
		case 11 : diff2 =  3 ; break;
		case 12 : diff2 =  4 ; break;
		case 13 : diff2 =  5 ; break;
		case 16 : diff2 =  4 ; break;
		case 17 : diff2 =  5 ; break;
		case 18 : diff2 =  6 ; break;
		case 22 : diff2 =  6 ; break;
		case 23 : diff2 =  7 ; break;
		case 24 : diff2 =  8 ; break;
		case 27 : diff2 =  7 ; break;
		case 28 : diff2 =  8 ; break;
		case 29 : diff2 =  9 ; break;
		case 30 : diff2 = 10 ; break;
		case 33 : diff2 =  9 ; break;
		case 34 : diff2 = 10 ; break;
		case 35 : diff2 = 11 ; break;
		case 36 : diff2 = 12 ; break;
		case 39 : diff2 = 11 ; break;
		case 40 : diff2 = 12 ; break;
	}

	if ( asc )
		diff2 = 12*octaves + diff2;
	else
		diff2 = -12*octaves - diff2;

	/*
	cout << "diff2  : " << diff2 << endl;
	cout << endl;
	*/

	if ( abs ( diff1 - diff2 ) < 3 ) //why 3 (minor third)?
		return 1.0;
	else
		return -1.0;

}


} /* namespace musical */
