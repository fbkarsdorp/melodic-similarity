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


#include "ISMIR2019ExactPitch40SimilarityRater.h"
#include "ISMIR2019Symbol.h"
#include "ISMIR2019Sequences.h"

namespace musical {

ISMIR2019ExactPitch40SimilarityRater::ISMIR2019ExactPitch40SimilarityRater() {
	// TODO Auto-generated constructor stub

}

ISMIR2019ExactPitch40SimilarityRater::~ISMIR2019ExactPitch40SimilarityRater() {
	// TODO Auto-generated destructor stub
}

double ISMIR2019ExactPitch40SimilarityRater::getScore(Sequences * const seqs, const int x1, const int y1, const int x2, const int y2) const {

	//for now ignore x1 and y1. Only return the similarity of the symbols associated with the destination cell

	ISMIR2019Symbol * s1 = static_cast<ISMIR2019Symbol *>(seqs->getSeq1()->getSymbolAt(x2));
	ISMIR2019Symbol * s2 = static_cast<ISMIR2019Symbol *>(seqs->getSeq2()->getSymbolAt(y2));

	//cout << x2 << " : " << pitch1 << " - " << y2 << " : " << pitch2 << endl;

	int pitchShift = static_cast<ISMIR2019Sequences *>(seqs)->getPitch40Shift();

	if ( s1->pitch40 == s2->pitch40+pitchShift )
		return 1.0;
	else
		return -1.0;

	return -1.0;

}


}
