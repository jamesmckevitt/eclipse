;+
; NAME:
; 
;	GREEK
;
; PURPOSE:
; 
;       This function returns the string needed to draw the specified
;       greek character using either the vector graphics font no. 4,
;       or PostScript font 9.
;
;       If (!d.name eq 'PS') and (!p.font eq 0), then the PostScript
;       font will be used.  Otherwise, the vector font will be used.
;
; CALLING SEQUENCE:
; 
;	Result = GREEK(Name)
;
; INPUTS:
; 
;       Name - String specifying the greek character name. Valid
;              inputs are:
;
;              alpha, beta, gamma, delta, epsilon, zeta, eta, theta
;              iota, kappa, lambda, mu, nu, xi, omicron, pi, rho,
;              sigma, tau, upsilon, phi, chi, psi, omega
;
;              Alpha, Beta, Gamma, Delta, Epsilon, Zeta, Eta, Theta
;              Iota, Kappa, Lambda, Mu, Nu, Xi, Omicron, Pi, Rho,
;              Sigma, Tau, Upsilon, Phi, Chi, Psi, Omega
;
;              Although not greek, the following characters are also
;              valid (but will only work with the 'default' font !3):
;
;              angstrom, Angstrom, degrees, plus_minus
;
; KEYWORDS:
;
;       FORCE_PS - Set to use PostScript font, regardless of the value
;                  of !d.name and !p.font.
;
;       PLAIN - Set to just return Name in plain text.
;
;       APPEND_FONT - Set to append the characters specifying a
;                     'default' font: !3. That is, if this keyword is
;                     set, then the command
;
;                     Result=GREEK(theta,/APPEND_FONT)
;
;                     will return the string
;
;                     '!9q!3' for PostScript and '!4h!3' for vector
;                     fonts.
; 
;
; OUTPUTS:
; 
;       Result - The string containing the specified greek character.
;
; EXAMPLE:
;
;	Result=GREEK(theta)
;	
;       In this case, Result='!9q' if !d.name is 'PS' and !p.font is
;       0; otherwise, Result='!4h'
;
; MODIFICATION HISTORY:
; 
; 	David L. Windt, Bell Labs, September 1998.
;       windt@bell-labs.com
;                    
;-

function greek,input,plain=plain,force_ps=force_ps,append_font=append_font

case 1 of
    ;; postscript:
    keyword_set(force_ps) or  $
      ((!d.name eq 'PS') and (!p.font eq 0)): begin 
        default_font='!3'
        case input of
            'alpha': name='!9a'
            'beta': name='!9b'
            'gamma': name='!9g'
            'delta': name='!9d'
            'epsilon': name='!9e'
            'zeta': name='!9z'
            'eta': name='!9h'
            'theta': name='!9q'
            'iota': name='!9i'
            'kappa': name='!9k'
            'lambda': name='!9l'
            'mu': name='!9m'
            'nu': name='!9n'
            'xi': name='!9x'
            'omicron': name='!9o'
            'pi': name='!9p'
            'rho': name='!9r'
            'sigma': name='!9s'
            'tau': name='!9t'
            'upsilon': name='!9u'
            'phi': name='!9f'
            'chi': name='!9c'
            'psi': name='!9y'
            'omega': name='!9w'
            'Alpha': name='!9A'
            'Beta': name='!9B'
            'Gamma': name='!9G'
            'Delta': name='!9D'
            'Epsilon': name='!9E'
            'Zeta': name='!9Z'
            'Eta': name='!9H'
            'Theta': name='!9Q'
            'Iota': name='!9I'
            'Kappa': name='!9K'
            'Lambda': name='!9L'
            'Mu': name='!9M'
            'Nu': name='!9N'
            'Xi': name='!9X'
            'Omicron': name='!9O'
            'Pi': name='!9P'
            'Rho': name='!9R'
            'Sigma': name='!9S'
            'Tau': name='!9T'
            'Upsilon': name='!9U'
            'Phi': name='!9F'
            'Chi': name='!9C'
            'Psi': name='!9Y'
            'Omega': name='!9W'
            'Angstroms': name=string(197b)
            'angstroms': name=string(229b)
            'plus_minus': name=string(177b)
            'degrees': name=string(176b)
            else: name=''
        endcase
    end

    ;; vector fonts:
    else: begin     
        default_font='!3'
        case input of 
            'alpha': name='!4a'
            'beta': name='!4b'
            'gamma': name='!4c'
            'delta': name='!4d'
            'epsilon': name='!4e'
            'zeta': name='!4f'
            'eta': name='!4g'
            'theta': name='!4h'
            'iota': name='!4i'
            'kappa': name='!4j'
            'lambda': name='!4k'
            'mu': name='!4l'
            'nu': name='!4m'
            'xi': name='!4n'
            'omicron': name='!4o'
            'pi': name='!4p'
            'rho': name='!4q'
            'sigma': name='!4r'
            'tau': name='!4s'
            'upsilon': name='!4t'
            'phi': name='!4u'
            'chi': name='!4v'
            'psi': name='!4w'
            'omega': name='!4x'
            'Alpha': name='!4A'
            'Beta': name='!4B'
            'Gamma': name='!4C'
            'Delta': name='!4D'
            'Epsilon': name='!4E'
            'Zeta': name='!4F'
            'Eta': name='!4G'
            'Theta': name='!4H'
            'Iota': name='!4I'
            'Kappa': name='!4J'
            'Lambda': name='!4K'
            'Mu': name='!4L'
            'Nu': name='!4M'
            'Xi': name='!4N'
            'Omicron': name='!4O'
            'Pi': name='!4P'
            'Rho': name='!4Q'
            'Sigma': name='!4R'
            'Tau': name='!4S'
            'Upsilon': name='!4T'
            'Phi': name='!4U'
            'Chi': name='!4V'
            'Psi': name='!4W'
            'Omega': name='!4X'
            'Angstroms': name=string(197b)
            'angstroms': name=string(229b)
            'plus_minus': name=string(177b)
            'degrees': name=string(176b)
           else: name=''
        endcase
    end
endcase

if keyword_set(plain) then return,input else begin
    if keyword_set(append_font) then $
      return,name+default_font else $
      return,name
endelse
end
