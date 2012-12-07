/*
 * Adapted to work with FTH:
 *
 * Copyright (c) 2004-2012 Michael Scholz <mi-scholz@users.sourceforge.net>
 *
 * This file is part of FTH.
 *
 */

/*
**	ficllocal.h
**
** Put all local settings here.  This file will always ship empty.
**
*/

#if !defined(_FICLLOCAL_H_)
#define _FICLLOCAL_H_

#include "fth-config.h"

typedef char	ficlInteger8;
typedef unsigned char ficlUnsigned8;
typedef short	ficlInteger16;
typedef unsigned short ficlUnsigned16;
typedef int	ficlInteger32;
typedef unsigned int ficlUnsigned32;
#if (FTH_SIZEOF_LONG == 4)
typedef long long ficlInteger64;
typedef unsigned long long ficlUnsigned64;
#else
typedef long	ficlInteger64;
typedef unsigned long ficlUnsigned64;
#endif

/* It's not a pointer but the base for type FTH. */
#if (FTH_SIZEOF_LONG == FTH_SIZEOF_VOID_P)
typedef unsigned long ficlPointer;
typedef long	ficlSignedPointer;
#elif (FTH_SIZEOF_LONG_LONG == FTH_SIZEOF_VOID_P)
typedef unsigned long long ficlPointer;
typedef long long ficlSignedPointer;
#else
#error *** FTH requires sizeof(void*) == (sizeof(long) || sizeof(long long))
#endif

typedef long ficlInteger;
typedef unsigned long ficlUnsigned;
typedef ficlInteger64 ficl2Integer;
typedef ficlUnsigned64 ficl2Unsigned;
typedef double ficlFloat;
typedef ficlPointer FTH;
typedef ficlSignedPointer SIGNED_FTH;

#if defined(HAVE_STDBOOL_H)
#include <stdbool.h>
#else
#if !defined(HAVE__BOOL)
#if defined(__cplusplus)
typedef bool	_Bool;
#else
typedef unsigned char _Bool;
#endif
#endif
#if !defined(true)
#define bool 		_Bool
#define false 		0
#define true 		1
#define __bool_true_false_are_defined 1
#endif
#endif

#define FICL_FORTH_NAME			FTH_PACKAGE_NAME
#define FICL_FORTH_VERSION		FTH_PACKAGE_VERSION
#define FICL_PLATFORM_BASIC_TYPES	1
#define FICL_DEFAULT_DICTIONARY_SIZE	(1024 * 1024)
#define FICL_DEFAULT_STACK_SIZE		(1024 * 8)
#define FICL_DEFAULT_RETURN_SIZE	1024
#define FICL_DEFAULT_ENVIRONMENT_SIZE	(1024 * 8)
#define FICL_MIN_DICTIONARY_SIZE	(1024 * 512)
#define FICL_MIN_STACK_SIZE		512
#define FICL_MIN_RETURN_SIZE		512
#define FICL_MIN_ENVIRONMENT_SIZE	(1024 * 4)
#define FICL_MAX_LOCALS			2048
#define FICL_MAX_WORDLISTS		32
#define FICL_MAX_PARSE_STEPS		16
#define FICL_PAD_SIZE			1024
#define FICL_NAME_LENGTH		256
#define FICL_HASH_SIZE			241
#define FICL_USER_CELLS			1024
#define FICL_PLATFORM_ALIGNMENT		FTH_ALIGNOF_VOID_P
#define FICL_PLATFORM_EXTERN		/* empty */

#define FICL_PRIMITIVE_SET(Dict, Name, Code, Type)			\
	ficlDictionaryAppendPrimitive(Dict, Name, Code, (ficlUnsigned)(Type))

#define FICL_PRIM(Dict, Name, Code)					\
	FICL_PRIMITIVE_SET(Dict, Name, Code, FICL_WORD_DEFAULT)

#define FICL_PRIM_IM(Dict, Name, Code)					\
	FICL_PRIMITIVE_SET(Dict, Name, Code, FICL_WORD_IMMEDIATE)

#define FICL_PRIM_CO(Dict, Name, Code)					\
	FICL_PRIMITIVE_SET(Dict, Name, Code, FICL_WORD_COMPILE_ONLY)

#define FICL_PRIM_CO_IM(Dict, Name, Code)				\
	FICL_PRIMITIVE_SET(Dict, Name, Code, FICL_WORD_COMPILE_ONLY_IMMEDIATE)

#define FICL_PRIMITIVE_DOC_SET(Dict, Name, Code, Type, Docs) do {	\
	ficlWord *word;							\
									\
	word = ficlDictionaryAppendPrimitive(Dict, Name, Code,		\
	    (ficlUnsigned)(Type));					\
									\
	fth_word_doc_set(word, Docs);					\
} while (0)

#define FICL_PRIM_DOC(Dict, Name, Code)					\
	FICL_PRIMITIVE_DOC_SET(Dict, Name, Code,			\
	    FICL_WORD_DEFAULT, h_ ## Code)
#define FICL_PRIM_IM_DOC(Dict, Name, Code)				\
	FICL_PRIMITIVE_DOC_SET(Dict, Name, Code,			\
	    FICL_WORD_IMMEDIATE, h_ ## Code)
#define FICL_PRIM_CO_DOC(Dict, Name, Code)				\
	FICL_PRIMITIVE_DOC_SET(Dict, Name, Code,			\
	    FICL_WORD_COMPILE_ONLY, h_ ## Code)
#define FICL_PRIM_CO_IM_DOC(Dict, Name, Code)				\
	FICL_PRIMITIVE_DOC_SET(Dict, Name, Code,			\
	    FICL_WORD_COMPILE_ONLY_IMMEDIATE, h_ ## Code)

#endif				/* _FICLLOCAL_H_ */

/* end of ficllocal.h */
