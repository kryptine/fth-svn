#! /usr/opt/bin/fth -vdDs
\ examples/scripts/xg.fth.  Generated from xg.fth.in by configure.
\ 
\ xg.fth -- example from libxm.html 
\ (see ftp://ccrma-ftp.stanford.edu/pub/Lisp/libxm.tar.gz)
\
\ usage: ./xg.fth -display :0.0

dl-load libxg Init_libxg

1 #( "xg.fth" ) Fgtk_init drop

FGTK_WINDOW_TOPLEVEL Fgtk_window_new value window
window "delete_event" lambda: <{ w e data -- }> Fgtk_main_quit drop #t ; Fg_signal_connect drop
window "destroy" lambda: <{ w data -- }> Fgtk_main_quit ; Fg_signal_connect drop
window FGTK_CONTAINER 10 Fgtk_container_set_border_width drop

"Click Me!" Fgtk_button_new_with_label value button
button "clicked" lambda: <{ w data -- }>
  "\\ widget: %p\n" #( w )      fth-print
  "\\   data: %p\n" #( data )   fth-print
  "\\ *argv*: %s\n" #( *argv* ) fth-print
  #t
; window Fg_signal_connect drop

window FGTK_CONTAINER button Fgtk_container_add drop
button Fgtk_widget_show drop
window Fgtk_widget_show drop
Fgtk_main drop

\ xg.fth ends here
