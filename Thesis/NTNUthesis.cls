\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{NTNUthesis}

\LoadClass[a4paper,12pt]{report}

% Required packages
\RequirePackage[top=1in,bottom=1in]{geometry}
\RequirePackage{indentfirst}
\RequirePackage{setspace}
\RequirePackage{CJKutf8}
% \RequirePackage{xeCJK}
\RequirePackage{tocbibind}
\RequirePackage{hyperref}
\RequirePackage{tikz}
\RequirePackage[encoding=UTF8]{zhnumber}

% Options
\DeclareOption{master}{\gdef\@degreeZH{碩}}
\DeclareOption{doctor}{\gdef\@degreeZH{博}}
\DeclareOption{nowatermark}{\gdef\nowatermark{1}}
\ProcessOptions\relax

% Watermark of NTNU
% \ifdefined \nowatermark \else
%     \AtBeginShipout{
%         \AtBeginShipoutAddToBox{
%             \begin{tikzpicture}[overlay,remember picture]
%             \node[opacity=0.1] at (current page.center){
%                 \includegraphics[height=100pt,width=100pt]{NTNUgrey.png}
%             };
%             \end{tikzpicture}
%         }
%     }
% \fi

% Default spacing
\setstretch{1.8}


% Declared variables
\newcommand{\titleZH}[1]{\gdef\@titleZH{#1}}
\newcommand{\titleEN}[1]{\gdef\@titleEN{#1}}
\newcommand{\studentZH}[1]{\gdef\@studentZH{#1}}
\newcommand{\studentEN}[1]{\gdef\@studentEN{#1}}
\newcommand{\advisorZH}[1]{\gdef\@advisorZH{#1}}
\newcommand{\advisorEN}[1]{\gdef\@advisorEN{#1}}
\newcommand{\yearZH}[1]{\gdef\@yearZH{#1}}
\newcommand{\monthZH}[1]{\gdef\@monthZH{#1}}
\newcommand{\yearEN}[1]{\gdef\@yearEN{#1}}
\newcommand{\monthEN}[1]{\gdef\@monthEN{#1}}

% Customized commands
\newcommand{\makecover}{
\begin{titlepage}
    \begin{center}
    \begin{CJK*}{UTF8}{bkai}

    \begin{spacing}{1}
        \Large
        國~~立~~臺~~灣~~師~~範~~大~~學\\
        理~~學~~院\\
        地~~球~~科~~學~~系\\
        \@degreeZH~~士~~論~~文\\
        Department of Earth Sciences\\
        College of Science\\
        National Taiwan Normal University\\
        Master's Thesis
    \end{spacing}
    \vspace{5.5\baselineskip}

    \begin{spacing}{1}
    \LARGE
    \@titleZH\\
    \@titleEN
    \end{spacing}
    \vfill

    \begin{spacing}{1}
    \Large
    \@studentZH\\
    \@studentEN
    \end{spacing}
    \vspace{1.5\baselineskip}

    \begin{spacing}{1}
    \Large
    指導教授：\@advisorZH\\
    Advisor: \@advisorEN
    \end{spacing}
    \vspace{1.5\baselineskip}

    \begin{spacing}{1}
    \Large
    中~~華~~民~~國~~\zhdigits{\@yearZH}~~年~~\zhdigits{\@monthZH}~~月
    \end{spacing}

    \begin{spacing}{1}
    \Large
    {\@monthEN}~{\@yearEN}
    \end{spacing}

    \end{CJK*}
    \end{center}
\end{titlepage}
}

\newcommand{\maketoc}{
    \tableofcontents
    \clearpage
    \listoffigures
    \listoftables
    \clearpage
}

\newcommand{\makeenv}[3]{
    \newenvironment{#1}{
        \phantomsection
        \section*{\centerline{#2}}
        \addcontentsline{toc}{chapter}{#3}
    }{
        \clearpage
    }
}

\newcommand{\makeenvZH}[3]{
    \newenvironment{#1}{
        \phantomsection
        \section*{\centerline{#2}}
        \addcontentsline{toc}{chapter}{#3}
    }{
        \clearpage
    }
}

% Customized environments
\makeenvZH{abstractZH}{摘要}{Abstract (Chinese)}
\makeenvZH{acknowledgementsZH}{致謝}{Acknowledgements (Chinese)}
\makeenv{abstractEN}{Abstract}{Abstract}
\makeenv{acknowledgementsEN}{Acknowledgements}{Acknowledgements}
