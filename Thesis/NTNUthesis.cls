\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{NTNUthesis}

\LoadClass[a4paper,12pt]{report}

% Required packages
\RequirePackage[
  top=2.5cm,
  bottom=2.5cm,
  left=2.2cm,
  right=2.2cm
]{geometry}
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
\setstretch{1.5}

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
% --- Cover font helpers (NTNU cover spec) ---
\newcommand{\CJK十八}{\fontsize{18pt}{27pt}\selectfont}  % 18pt, 1.5 line height
\newcommand{\EN十四}{\fontsize{14pt}{21pt}\selectfont}   % 14pt, 1.5 line height
\newcommand{\EN十六}{\fontsize{16pt}{24pt}\selectfont}   % 16pt, 1.5 line height


% Customized commands
\newcommand{\makecover}{
\begin{titlepage}
    \begin{center}
    \begin{CJK*}{UTF8}{bkai}

    {\CJK十八
    國~~立~~臺~~灣~~師~~範~~大~~學\\
    理~~學~~院\\
    地~~球~~科~~學~~系\\
    \@degreeZH~~士~~論~~文\\
    }
    {\EN十四
    Department of Earth Sciences\\
    College of Science\\
    }
    {\EN十六
    National Taiwan Normal University\\
    Master's Thesis\\
    }

    \vfill

    {\CJK十八
    \@titleZH\\
    }
    {\fontsize{18pt}{27pt}\selectfont
    \@titleEN
    }

    \vfill

    {\CJK十八
    \@studentZH\\
    }
    {\fontsize{18pt}{27pt}\selectfont
    \@studentEN
    }
    \vfill

    {\CJK十八
    \@advisorZH\\
    }
    {\fontsize{18pt}{27pt}\selectfont
    \@advisorEN
    }
    \vfill

    {\CJK十八
    中~~華~~民~~國~~\zhdigits{\@yearZH}~~年~~\zhdigits{\@monthZH}~~月
    }

    {\fontsize{18pt}{27pt}\selectfont
    \@monthEN~\@yearEN
    }

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
\makeenvZH{abstractZH}{摘要}{摘要}
\makeenvZH{acknowledgementsZH}{致謝}{致謝}
\makeenv{abstractEN}{Abstract}{Abstract}
\makeenv{acknowledgementsEN}{Acknowledgements}{Acknowledgements}
