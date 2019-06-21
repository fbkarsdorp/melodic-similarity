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

#ifndef ISMIR2019MIDIINTERVALSIMILARITYRATER_H_
#define ISMIR2019MIDIINTERVALSIMILARITYRATER_H_

#include <SimilarityRater.h>

namespace musical {

/**
 * Calculates similarity between a ISMIR2019-symbol and a Midi-symbol
 * Only the pitch information can be used.
 */
class ISMIR2019MidiIntervalSimilarityRater: public musical::SimilarityRater {
public:

	/**
	 * Constructor
	 */
	ISMIR2019MidiIntervalSimilarityRater();

	/**
	 * Destructor
	 */
	virtual ~ISMIR2019MidiIntervalSimilarityRater();

	/**
	 * Returns the score of aligning Symbols x2 with y2, where x2 and y2 are indices in seq1 and seq2.
	 * For now, x1 and y1 are ignored.
	 * seq1 should be a Midi Sequence, seq2 should be a ISMIR2019-Sequence !!
	 *
	 */
	double getScore(Sequences * const seqs, const int x1, const int y1, const int x2, const int y2) const;

	virtual std::string getName() { return "ISMIR2019MidiIntervalSimilarityRater"; };

};

} /* namespace musical */
#endif /* ISMIR2019MIDIINTERVALSIMILARITYRATER_H_ */
