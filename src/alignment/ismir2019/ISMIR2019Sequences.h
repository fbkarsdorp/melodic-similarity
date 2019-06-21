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


#ifndef ISMIR2019SEQUENCES_H_
#define ISMIR2019SEQUENCES_H_

#include "ISMIR2019Sequence.h"
#include "Sequences.h"
#include <map>
#include <utility>

namespace musical {

/**
 * Pair of two sequences of ISMIR2019Symbol.
 */
class ISMIR2019Sequences: public musical::Sequences {

private:
	/**
	 * Constructor
	 */
	ISMIR2019Sequences(); //private to ensure that only ISMIR2019Sequences are assigned

public:
	/**
	 * Constructor
	 * seq1 : pointer to first sequence
	 * seq2 : pointer to second sequence
	 */
	ISMIR2019Sequences(ISMIR2019Sequence * seq1, ISMIR2019Sequence * seq2) : Sequences(seq1,seq2) { computePitch40HistogramShift(); };

	/**
	 * Destructor
	 */
	virtual ~ISMIR2019Sequences();

	/**
	 * Computes how much the base40 pitch of seq2 has to be shifted in order to correspond to the pitch of seq1.
	 * This is computed by finding the shift of the normalized pitch histograms of seq1 and seq2 that results
	 * in highest histogram intersection.
	 */
	int getPitch40Shift() const { return pitch40HistogramShift; };

	/**
	 * Get the n^th pitchshift
	 * n = 0 : pitch shift with highest pitch histogram intersection
	 * n = 1 : pitch shift with second highest pitch histogram intersecion
	 * etc
	 * if n > number of found intersections: return 0
	 */
	int getNthComputedPitch40Shift(unsigned int n) const {
		if (n >= intersections.size() ) {
			std::cerr << "Warning: Number of computed pitchhistogram intersection is smaller than " << n << "." << std::endl;
			return 0;
		}
		return intersections[intersections.size()-1-n].first;
	}

	/**
	 * Provide a pitch shift
	 */
	void setPitch40Shift(int s) { pitch40HistogramShift = s; }

	/**
	 * Get the pitchshift in base-12 representation
	 */
	int getPitch12Shift() const;

private:
	/**
	 * The interval that has to be added to the pitches of seq2 to correspond with seq1
	 */
	int pitch40HistogramShift;

	/**
	 * All intervals to to be added to the pitches of seq2 to correspond with seq1, ordered in ascending order of histogram intersection
	 */
	std::vector<std::pair<int,double> > intersections; // (shift, intersection) pairs in descending order of intersection

	/**
	 * Computes the interval that has to be added to the pitches of seq2 to correspond with seq1
	 */
	void computePitch40HistogramShift();

};

}

#endif /* ISMIR2019SEQUENCES_H_ */
