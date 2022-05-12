(require 'assemble)

(defvar thesis-msc/out-dir "build")

(setq assemble-outdir thesis-msc/out-dir)

(setq thesis-msc/tex-files
      '(title-page
	intro
	background
	evolution
	approx-dp
	deicide
	appendix-stochastic
	appendix-analysis
	appendix-measure
	conclusion))

(setq thesis-msc/targets
      (mapcar (lambda (file) (intern (concat (symbol-name file) ".tex")))
	      thesis-msc/tex-files))

(defun default ()
  (assemble-ifchanged "__dummy__" 'thesis.pdf))

(assemble-target "thesis.pdf" depending on (cons 'init thesis-msc/targets)
		 (latex-build "thesis" thesis-msc/out-dir t))

(assemble-wildcard "pdf" depending on '(_tex)
		   (latex-build _filename thesis-msc/out-dir t))

(assemble-target "background.tex" depending on '(build/graphics/brownian.pdf
						 build/graphics/munos-value.pdf))

(assemble-target "build/graphics/brownian.pdf" depending on '(graphics/brownian.tex)
		 (latex-build "graphics/brownian" thesis-msc/out-dir nil))

(assemble-target "build/graphics/munos-value.pdf" depending on '(graphics/munos-value.tex
								 graphics/munos-value.dat)
		 (latex-build "graphics/munos-value" thesis-msc/out-dir nil))

(assemble-target "approx-dp.tex" depending on '(build/graphics/functor.pdf
						build/graphics/representations.pdf
						build/graphics/discretized-trajectory.pdf))

(assemble-target "build/graphics/functor.pdf" depending on '(graphics/functor.tex)
		 (latex-build "graphics/functor" thesis-msc/out-dir nil))

(assemble-target "build/graphics/representations.pdf"
		 depending on '(graphics/representations.tex
				graphics/representations-pdf.dat
				graphics/representations-mean.dat
				graphics/representations-cat.dat
				graphics/representations-quantile.dat)
		 (latex-build "graphics/representations" thesis-msc/out-dir nil))

(assemble-target "build/graphics/discretized-trajectory.pdf" depending on '(graphics/discretized-trajectory.tex)
		 (latex-build "graphics/discretized-trajectory" thesis-msc/out-dir nil))

(defun latex-build (file-name-base &optional out-dir bibtex)
  (let* ((inner-dir (if (string-match-p "/" file-name-base)
			(car (split-string file-name-base "/"))
		      ""))
	 (full-out-dir
	  (if (string-empty-p inner-dir) out-dir (concat out-dir "/" inner-dir)))
	 (fname (file-name-base file-name-base))
	 (pdflatex-flags (and out-dir (concat "-output-directory=" full-out-dir)))
	 (pdflatex-cmd (concat "pdflatex " pdflatex-flags " " file-name-base ".tex"))
	 (bibtex-cmd (concat "bibtex " (and out-dir (concat full-out-dir "/")) fname))
	 (cmds (if bibtex
		   (list pdflatex-cmd bibtex-cmd pdflatex-cmd pdflatex-cmd)
		 (list pdflatex-cmd))))
    (mapcar (lambda (cmd) (shell-command-to-string cmd)) cmds)))
    ;; (mapcar (lambda (cmd) (message (shell-command-to-string cmd))) cmds)))

(defun init ()
  (shell-command (concat "mkdir -p " thesis-msc/out-dir))
  (shell-command (concat "mkdir -p " thesis-msc/out-dir "/graphics"))
  (shell-command (concat "cp *.tex " thesis-msc/out-dir))
  (shell-command (concat "cp -R graphics " thesis-msc/out-dir "/graphics")))
