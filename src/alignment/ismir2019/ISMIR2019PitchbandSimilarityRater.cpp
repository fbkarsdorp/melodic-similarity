/*
Author: Peter van Kranenburg (peter.van.kranenburg@meertens.knaw.nl)
Copyright 2011 Meertens Institute (KNAW)

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


#include <cmath>

#include "ISMIR2019PitchbandSimilarityRater.h"
#include "ISMIR2019Symbol.h"
#include "ISMIR2019Sequences.h"

namespace musical {

double ISMIR2019PitchbandSimilarityRater::getScore(Sequences * seqs, int x1, int y1, int x2, int y2) const {

	ISMIR2019Symbol * s1;
	ISMIR2019Symbol * s2;
	double result;
	int diff;

	//for now ignore x1 and y1. Only return the similarity of the symbols associated with the destination cell

	//dynamic_cast would be better, but much, much slower
	s1 = static_cast<ISMIR2019Symbol *>(seqs->getSeq1()->getSymbolAt(x2));
	s2 = static_cast<ISMIR2019Symbol *>(seqs->getSeq2()->getSymbolAt(y2));
	//s1 = static_cast<ISMIR2019Symbol *>(seqs->seq1->symbols[x2]);
	//s2 = static_cast<ISMIR2019Symbol *>(seqs->seq2->symbols[y2]);

	int pitchshift = static_cast<ISMIR2019Sequences *>(seqs)->getPitch40Shift();

	//cout << x2 << " : " << pitch1 << " - " << y2 << " : " << pitch2 << endl;

	result = 0.0;

	int im1 = abs (s1->pitch40 - (s2->pitch40+pitchshift));
	diff = im1 % 40;
	if ( diff > 23 ) result = -1.0; else result = 1.0 - ( (double)diff * 1.0/23.0 );

	return result;

}

}
