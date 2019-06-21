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


#ifndef ISMIR2019SYMBOL_H_
#define ISMIR2019SYMBOL_H_

#include "Symbol.h"

#include <string>

namespace musical {

/**
 * Represents an "ISMIR2019Symbol"
 * has: pitch40, IMA and phrasepos.
 */
class ISMIR2019Symbol: public musical::Symbol {
public:
	/**
	 * Constructor
	 */
	ISMIR2019Symbol();

	/**
	 * Destructor
	 */
	virtual ~ISMIR2019Symbol();

	/**
	 * Returns a float representation of the symbol
	 */
	virtual float toFloat() const { return (float)pitch40; }

	/**
	 * Returns a string representation of the symbol
	 */
	std::string toString() const ;

	std::string contour3;
	int diatonicinterval;
	std::string tonic;
	int scaledegree;
	float phrasepos;
	std::string pitch;
	std::string imacontour;
	std::string beat_fraction_str;
    float beat;
    float imaweight;
    float duration;
    int chromaticinterval;
    std::string mode;
    int midipitch;
    int pitch12;
    float IOR;
    std::string metriccontour;
    std::string timesignature;
    std::string contour5;
    std::string beat_str;
    int pitch40;
    float beatstrength;
};

}

#endif /* ISMIR2019SYMBOL_H_ */
