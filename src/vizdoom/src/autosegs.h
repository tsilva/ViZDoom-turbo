/*
** autosegs.h
** Arrays built at link-time
**
**---------------------------------------------------------------------------
** Copyright 1998-2006 Randy Heit
** All rights reserved.
**
** Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions
** are met:
**
** 1. Redistributions of source code must retain the above copyright
**    notice, this list of conditions and the following disclaimer.
** 2. Redistributions in binary form must reproduce the above copyright
**    notice, this list of conditions and the following disclaimer in the
**    documentation and/or other materials provided with the distribution.
** 3. The name of the author may not be used to endorse or promote products
**    derived from this software without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
** IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
** OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
** IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
** INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
** NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
** THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**---------------------------------------------------------------------------
**
*/

#ifndef AUTOSEGS_H
#define AUTOSEGS_H

#include "doomtype.h"

#define REGMARKER(x) (x)
typedef void *REGINFO;

// List of Action functons
extern REGINFO ARegHead;
extern REGINFO ARegTail;

// List of TypeInfos
extern REGINFO CRegHead;
extern REGINFO CRegTail;

// List of properties
extern REGINFO GRegHead;
extern REGINFO GRegTail;

// List of variables
extern REGINFO MRegHead;
extern REGINFO MRegTail;

// List of MAPINFO map options
extern REGINFO YRegHead;
extern REGINFO YRegTail;

//VIZDOOM_CODE
#if defined(__GNUC__) && defined(__ELF__)
extern "C"
{
	extern REGINFO __start_areg[];
	extern REGINFO __stop_areg[];
	extern REGINFO __start_creg[];
	extern REGINFO __stop_creg[];
	extern REGINFO __start_greg[];
	extern REGINFO __stop_greg[];
	extern REGINFO __start_mreg[];
	extern REGINFO __stop_mreg[];
	extern REGINFO __start_yreg[];
	extern REGINFO __stop_yreg[];
}
#endif

//VIZDOOM_CODE
static inline REGINFO *VIZ_AutoSegStart(REGINFO &head, REGINFO &tail)
{
#if defined(__GNUC__) && defined(__ELF__)
	if (&head == &ARegHead) return __start_areg;
	if (&head == &CRegHead) return __start_creg;
	if (&head == &GRegHead) return __start_greg;
	if (&head == &MRegHead) return __start_mreg;
	if (&head == &YRegHead) return __start_yreg;
#endif
	return &head < &tail ? &head : &tail;
}

//VIZDOOM_CODE
static inline REGINFO *VIZ_AutoSegStop(REGINFO &head, REGINFO &tail)
{
#if defined(__GNUC__) && defined(__ELF__)
	if (&head == &ARegHead) return __stop_areg;
	if (&head == &CRegHead) return __stop_creg;
	if (&head == &GRegHead) return __stop_greg;
	if (&head == &MRegHead) return __stop_mreg;
	if (&head == &YRegHead) return __stop_yreg;
#endif
	return (&head < &tail ? &tail : &head) + 1;
}

class FAutoSegIterator
{
	public:
		FAutoSegIterator(REGINFO &head, REGINFO &tail)
		{
			Head = VIZ_AutoSegStart(head, tail); //VIZDOOM_CODE
			Tail = VIZ_AutoSegStop(head, tail); //VIZDOOM_CODE
			Probe = Head;
		}
		REGINFO operator*() const NO_SANITIZE
		{
			return Probe < Tail ? *Probe : NULL; //VIZDOOM_CODE
		}
		FAutoSegIterator &operator++() NO_SANITIZE
		{
			do
			{
				++Probe;
			} while (Probe < Tail && *Probe == 0); //VIZDOOM_CODE
			return *this;
		}
		void Reset()
		{
			Probe = Head;
		}

	protected:
		REGINFO *Probe;
		REGINFO *Head;
		REGINFO *Tail;
};

#endif
