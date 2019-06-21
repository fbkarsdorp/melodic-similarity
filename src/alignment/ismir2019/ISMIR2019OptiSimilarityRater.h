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


#ifndef ISMIR2019OPTISIMILARITYRATER_H_
#define ISMIR2019OPTISIMILARITYRATER_H_

#include "SimilarityRater.h"
#include "ISMIR2019Symbol.h"

namespace musical {

/**
 * Computes the substitution score for OptiSymbols.
 */
class ISMIR2019OptiSimilarityRater: public musical::SimilarityRater {
public:
	/**
	 * Constructor
	 */
	ISMIR2019OptiSimilarityRater() { };

	/**
	 * Destructor
	 */
	virtual ~ISMIR2019OptiSimilarityRater() { };

	/**
	 * Computes the similarity of seq1[x1] and seq2[y2].
	 * x1 and y1 are ignored for now.
	 */
	virtual double getScore(Sequences * const seqs, const int x1, const int y1, const int x2, const int y2) const;

	virtual std::string getName() { return "ISMIR2019OptiSimilarityRater"; };

};

}

#endif /* ISMIR2019OPTISIMILARITYRATER_H_ */
