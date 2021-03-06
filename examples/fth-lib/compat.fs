\ -*- snd-forth -*-
\ === begin of gforth/0.6.2/compat/defer.fs ===
\ deferred words and perform

: /string ( c-addr1 u1 n -- c-addr2 u2 ) tuck - >r + r> dup 0< if - 0 then ;

\ This file is in the public domain. NO WARRANTY.

: noop ;

: perform ( ? addr -- ? )
    @ execute ;

: defer ( "name" -- )
    create ['] noop , \ you should not rely on initialization with noop
does> ( ? -- ? )
    perform ;

: is ( xt "name" -- )
    ' >body ! ;

: [is] ( compilation: "name" -- ; run-time: xt -- )
    ' >body postpone Literal postpone ! ; immediate
\ === end of gforth/0.6.2/compat/defer.fs ===

\ === begin of gforth/0.6.2/compat/loops.fs ===
\ +DO, -DO...-LOOP and friends

\ This file is in the public domain. NO WARRANTY.

\ Hmm, this would be a good application for ]] ... [[

\ The program uses the following words
\ from CORE :
\ : POSTPONE over min ; immediate 2dup IF swap THEN drop negate +LOOP
\ ELSE 2drop < 1+ DO u<
\ from CORE-EXT :
\ ?DO u> 
\ from BLOCK-EXT :
\ \ 
\ from FILE :
\ ( 

: +DO ( compile-time: -- do-sys; run-time: n1 n2 -- )
    POSTPONE over POSTPONE min POSTPONE ?do ; immediate

: umin ( u1 u2 -- u )
    2dup u>
    IF
	swap
    THEN
    drop ;

: U+DO ( compile-time: -- do-sys; run-time: u1 u2 -- )
    POSTPONE over POSTPONE umin POSTPONE ?do ; immediate

\ -DO...-LOOP

\ You have to use the -LOOP implemented below with -DO or U-DO, you
\ cannot use it with ?DO

\ The implementation is a little more complicated. Basically, we
\ create an IF DO ... +LOOP THEN structure. The DO..+LOOP does not
\ exhibit the anomaly of ?DO...+LOOP; the IF..THEN is needed to
\ correct for DO's at-least-once semantics. The parameters are
\ conditioned a bit such that the result is as expected.

\ I define a '-do-sys' (whose implementation is 'orig do-sys'). Like
\ ANS Forth loop structures, this implementation of -DO..-LOOP
\ cannot be mixed with any other structures.

\ unlike Gforth's -LOOP, this implementation cannot handle all
\ unsigned increments, only positive integers
: -LOOP ( compilation -do-sys -- ; run-time loop-sys1 +n -- | loop-sys2 )
    POSTPONE negate POSTPONE +loop
    POSTPONE else POSTPONE 2drop POSTPONE then ; immediate

: -DO ( compilation -- -do-sys ; run-time n1 n2 -- | loop-sys )
    POSTPONE 2dup POSTPONE < POSTPONE if
    POSTPONE swap POSTPONE 1+ POSTPONE swap POSTPONE do ; immediate

: U-DO ( compilation -- -do-sys ; run-time u1 u2 -- | loop-sys )
    POSTPONE 2dup POSTPONE u< POSTPONE if
    POSTPONE swap POSTPONE 1+ POSTPONE swap POSTPONE do ; immediate
\ === end of gforth/0.6.2/compat/loops.fs ===

\ === begin of gforth/0.6.2/compat/control.fs
: ?DUP-IF ( compilation -- orig ; run-time n -- n| ) \ gforth	question-dupe-if
    POSTPONE ?dup POSTPONE if ; immediate

: ?DUP-0=-IF ( compilation -- orig ; run-time n -- n| ) \ gforth	question-dupe-zero-equals-if
    POSTPONE ?dup POSTPONE 0= POSTPONE if ; immediate
\ === end of gforth/0.6.2/compat/control.fs
  
\ compat.fs ends here
