
(defun dump-mnist-instance (name)
  (dump-loadable-structure name *mlp*)
  )

;; (defun draw (image-float-vector)
;;   (unless (boundp '*irtviewer*)
;;     (make-irtviewer)
;;     (send *irtviewer* :change-background #f(1 1 1)))
;;   (let* ((simage (instance color-image :init 28 28)))
;;     (send *irtviewer* :viewer :viewsurface :putimage simage :depth 24)
;;     ))
(defun draw-train-images ()
  (draw-mnist-images *train-images* *train-labels*))
(defun draw-test-images ()
  (draw-mnist-images *test-images* *test-labels*))

(defun draw-mnist-images (&optional (imgs *test-images*)
				    (labels *test-labels*)
				    (i 0))
  (catch :exit-draw
    (do-until-key
     (draw-test-image i imgs labels)
     (incf i)
     (unix::sleep 1)
     (if (>= i (length labels)) (throw :exit-draw nil))
     )))

(defun draw-test-image (i cnt y &optional (imgs *test-images*)
			  (labels *test-labels*))
  (let ((max0 0.0) (maxi0 -1) (max1 0.0) (maxi1 -1))
    (unless (boundp '*irtviewer*) (make-irtviewer))
    (dotimes (j (array-dimension y 1))
      (cond
       ((> (aref y 0 j) max0)
	(setq max0 (aref y 0 j) maxi0 j))
       ((> (aref y 0 j) max1)
	(setq max1 (aref y 0 j) maxi1 j))))
    (cond
     ((= maxi0 (round (elt (elt labels i) 0))) t)
     (t
      (format t ";; ~A:~A labels=~A" cnt i (elt labels i))
      (format t " maxi0=~A, p=~A , maxi1=~A, p=~A~%" maxi0 max0 maxi1 max1)
      (send *irtviewer* :draw-objects)
      (send *irtviewer* :string (* 28 8) 20
	    (format nil "cnt: ~A" cnt) x::font-timesb24)
      (send *irtviewer* :string (* 28 8) 50
	    (format nil "right: ~A" (round (elt (elt labels i) 0)))
	    x::font-timesb24)
      (send *irtviewer* :string (* 28 8) 80
	    (format nil "0: ~A pr: ~A " (round maxi0) max0)
	    x::font-timesb24)
      (send *irtviewer* :string (* 28 8) 110
	    (format nil "1: ~A pr: ~A " (round maxi1) max1)
	    x::font-timesb24)
      (draw-mnist-image (elt imgs i))
      nil))
    ))

(defun draw-mnist-image (v784image
			 &optional 
			 (vs (send *irtviewer* :viewer :viewsurface)))
  (let* ((imgv (coerce (scale 255 v784image) string))
	 (img (instance grayscale-image :init 28 28)))
    (setq (img . entity) imgv)
    (setq img (send img :double))
    (setq img (send img :double))
    (setq img (send img :double))
    (setq img (send img :to24))
    (send vs :putimage img :depth 24)
    ))

(in-package "X")

(defmethod irtviewer
  (:string (x y str &optional ;;(fid x::font-helvetica-bold-12))
	      (fid x::font-a14))
	   (send-all x::subwindows :2d-mode)
	   ;;(gl::glxUseXfont fid 32 96 (+ 1000 32))
	   (gl::glxUseXfont fid 32 96 (+ 1000 32))
	   (gl::glRasterPos2i (round x) (- (send self :height) (round y)))
	   (gl::glListBase 1000)
	   (gl::glCallLists (length str) gl::GL_UNSIGNED_BYTE str)
	   (send-all x::subwindows :3d-mode)
	   (send self :flush)
	   )
  )

(in-package "IMAGE")

(defmethod color-image
  (:halve
   (&optional (simage)) 	;512*512->256x256
   (let* ((swidth (/ (send self :width) 2))
	  (sheight (/ (send self :height) 2)))
     (if (null simage)
	 (setq simage (instance class :init swidth sheight)))
     ;;(halve-image self simage pixel-bytes)
     (halve-image self simage) ;; inaba 2018.11.16
     simage))
  (:double
   (&optional (simage)) 	;512*512->256x256
   (let* ((swidth (* (send self :width) 2))
	  (sheight (* (send self :height) 2)))
     (if (null simage)
	 (setq simage (instance class :init swidth sheight)))
     ;;(halve-image self simage pixel-bytes)
     (double-image self simage) ;; inaba 2018.11.16
     simage))
  (:convolve
   (&optional (c
	       (make-array '(3 3) :element-type :byte
			   :initial-contents '((1 0 -1) (1 0 -1) (1 0 -1)))))
   (let ((j (copy-object self)))
     (convolve3 c 1 self j)
     j))
  )

(in-package "USER")
