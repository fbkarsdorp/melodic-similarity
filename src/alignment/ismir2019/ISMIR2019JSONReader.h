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


#ifndef ISMIR2019JSONREADER_H_
#define ISMIR2019JSONREADER_H_

#include "JSONReader.h"
#include "Sequence.h"

namespace musical {

/**
 * Generates a sequence of ISMIR2019Symbol symbols from a JSON string
 */
class ISMIR2019JSONReader: public musical::JSONReader {
public:
	/**
	 * Constructor
	 */
	ISMIR2019JSONReader();

	/**
	 * Constructor
	 * s : pointer to the source of the json string
	 */
	ISMIR2019JSONReader(JSONSource * s) : JSONReader(s) {};

	/**
	 * Destructor
	 */
	virtual ~ISMIR2019JSONReader();

	/**
	 * Generate the sequence.
	 */
	virtual Sequence* generateSequence() const;
};

}

#endif /* ISMIR2019JSONREADER_H_ */
