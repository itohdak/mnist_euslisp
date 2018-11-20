;;
;;
;;
(defun mnist-read-a-2d-image
  (f width height
     &optional (img (make-array (list width height)
				:element-type :byte)))
  (catch :mnist-exit
    (let ((r nil) v)
      (dotimes (i height)
	(dotimes (j width)
	  (if (eq (setq v (read f nil r)) r)
	      (throw :mnist-exit nil)
	    (setf (aref img i j) v)))))
    img))

(defun mnist-read-2d-image
  (fname width height)
  (let (res img eof)
    (with-open-file
     (f fname :direction :input)
     (while (setq img (mnist-read-a-2d-image f width height))
       (push img res)))
    res))

(defun mnist-read-a-image
  (f length
     &optional (img (make-array length :element-type :byte)))
  (catch :mnist-exit
    (let ((r '(nil)) v)
      (dotimes (i length)
	(if (eq (setq v (read f nil r)) r)
	    (throw :mnist-exit nil)
	  (setf (aref img i) v))))
    img))

(defun mnist-read-image
  (fname length)
  (let (res img eof)
    (with-open-file
     (f fname :direction :input)
     (while (setq img (mnist-read-a-image f length))
       (push img res)))
    res))


(defun test-2d nil
  (setq *test-images* (mnist-read-2d-image "test-images.txt" 28 28))
  (format t "test-images:~A~%" (length *test-images*))
  (setq *test-labels* (mnist-read-2d-image "test-labels.txt" 1 1))
  (format t "test-labels:~A~%" (length *test-labels*))
  (setq *train-labels* (mnist-read-2d-image "train-labels.txt" 1 1))
  (format t "train-labels:~A~%" (length *train-labels*))
  (setq *train-images* (mnist-read-2d-image "train-images.txt" 28 28))
  (format t "train-images:~A~%" (length *train-images*))
  )

(defun test nil
  (setq *test-images* (mnist-read-image "test-images.txt" 784))
  (format t "test-images:~A~%" (length *test-images*))
  (setq *test-labels* (mnist-read-image "test-labels.txt" 1))
  (format t "test-labels:~A~%" (length *test-labels*))
  (setq *train-labels* (mnist-read-image "train-labels.txt" 1))
  (format t "train-labels:~A~%" (length *train-labels*))
  (setq *train-images* (mnist-read-image "train-images.txt" 784))
  (format t "train-images:~A~%" (length *train-images*))
  )


