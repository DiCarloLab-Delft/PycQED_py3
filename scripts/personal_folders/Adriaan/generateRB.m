function generateRB(maxClif, steps, file)
% "generateRB" generates a random sequence of Clifford in terms of rotations
%       on the Bloch sphere, add the "undo" gate, and save a file with the
%       id-Numbers of the rotation pulses defining the sequence
% NOTE: the number of Cliffords used per point are generated in a log scale
%
% INPUT
%   1) max number of Clifford to generate
%   2) number of steps (sequences)
%   3) name of the file to save the sequence


% generate Nbr of Cliffords in a log scale
normFactor = flip(-log(linspace(0,1,steps+1)));
normFactor = normFactor(1:end-1);
nbrClif = round(maxClif*normFactor/normFactor(end));

% nbrClif = [1:1:10]; % number of Cliffords
useSeed = 0; % 0 = random seed; 1 = use seed named 's'
saveFile = 1;
% file = 'RBlist.txt'; % name of the file

if useSeed == 0
    rng('shuffle'); % set the random generator
    s = rng; % store information of the seed
elseif useSeed == 1
    rng(s);
else
    fprintf('Use seed or not? \n')
end


% generate basic gates (and labels)
pauli = PauliMatrices(1);
labelR{1} = 'Id';
labelR{2} = 'X';
labelR{3} = 'Y';
labelR{4} = 'X90';
labelR{5} = 'Y90';
labelR{6} = 'Xm90';
labelR{7} = 'Ym90';

R{1} = eye(2);
R{2} = expm(-1i*pi*pauli{2}/2); % X
R{3} = expm(-1i*pi*pauli{3}/2); % Y
R{4} = expm(-1i*pi*pauli{2}/4); % X90
R{5} = expm(-1i*pi*pauli{3}/4); % Y90
R{6} = expm(1i*pi*pauli{2}/4); % Xm90
R{7} = expm(1i*pi*pauli{3}/4); % Ym90

% build the cliffords
C{1} = eye(2);
C{2} = R{strcmp(labelR,'X90')} * R{strcmp(labelR,'Y90')};
C{3} = R{strcmp(labelR,'Ym90')} * R{strcmp(labelR,'Xm90')};
C{4} = R{strcmp(labelR,'X')};
C{5} = R{strcmp(labelR,'Xm90')} * R{strcmp(labelR,'Ym90')};
C{6} = R{strcmp(labelR,'Ym90')} * R{strcmp(labelR,'X90')};
C{7} = R{strcmp(labelR,'Y')};
C{8} = R{strcmp(labelR,'X90')} * R{strcmp(labelR,'Ym90')};

C{9} = R{strcmp(labelR,'Y90')} * R{strcmp(labelR,'X90')};
C{10} = R{strcmp(labelR,'Y')} * R{strcmp(labelR,'X')};
C{11} = R{strcmp(labelR,'Xm90')} * R{strcmp(labelR,'Y90')};
C{12} = R{strcmp(labelR,'Y90')} * R{strcmp(labelR,'Xm90')};
C{13} = R{strcmp(labelR,'X')} * R{strcmp(labelR,'Y90')};
C{14} = R{strcmp(labelR,'Xm90')};
C{15} = R{strcmp(labelR,'Xm90')} * R{strcmp(labelR,'Ym90')} * R{strcmp(labelR,'X90')};
C{16} = R{strcmp(labelR,'Ym90')};

C{17} = R{strcmp(labelR,'X90')};
C{18} = R{strcmp(labelR,'X90')} * R{strcmp(labelR,'Y90')} * R{strcmp(labelR,'X90')};
C{19} = R{strcmp(labelR,'X')} * R{strcmp(labelR,'Ym90')};
C{20} = R{strcmp(labelR,'Y')} * R{strcmp(labelR,'X90')};
C{21} = R{strcmp(labelR,'X90')} * R{strcmp(labelR,'Ym90')} * R{strcmp(labelR,'X90')};
C{22} = R{strcmp(labelR,'Y90')};
C{23} = R{strcmp(labelR,'Y')} * R{strcmp(labelR,'Xm90')};
C{24} = R{strcmp(labelR,'Xm90')} * R{strcmp(labelR,'Y90')} * R{strcmp(labelR,'X90')};

% Write the Cliffords in terms of Pauli
CP{1} = 1;
CP{2} = [find(strcmp(labelR,'Y90')) find(strcmp(labelR,'X90'))];
CP{3} = [find(strcmp(labelR,'Xm90')) find(strcmp(labelR,'Ym90'))];
CP{4} = find(strcmp(labelR,'X'));
CP{5} = [find(strcmp(labelR,'Ym90')) find(strcmp(labelR,'Xm90'))];
CP{6} = [find(strcmp(labelR,'X90')) find(strcmp(labelR,'Ym90'))];
CP{7} = find(strcmp(labelR,'Y'));
CP{8} = [find(strcmp(labelR,'Ym90')) find(strcmp(labelR,'X90'))];

CP{9} = [find(strcmp(labelR,'X90')) find(strcmp(labelR,'Y90'))];
CP{10} = [find(strcmp(labelR,'X')) find(strcmp(labelR,'Y'))];
CP{11} = [find(strcmp(labelR,'Y90')) find(strcmp(labelR,'Xm90'))];
CP{12} = [find(strcmp(labelR,'Xm90')) find(strcmp(labelR,'Y90'))];
CP{13} = [find(strcmp(labelR,'Y90')) find(strcmp(labelR,'X'))];
CP{14} = find(strcmp(labelR,'Xm90'));
CP{15} = [find(strcmp(labelR,'X90')) find(strcmp(labelR,'Ym90')) find(strcmp(labelR,'Xm90'))];
CP{16} = find(strcmp(labelR,'Ym90'));

CP{17} = find(strcmp(labelR,'X90'));
CP{18} = [find(strcmp(labelR,'X90')) find(strcmp(labelR,'Y90')) find(strcmp(labelR,'X90'))];
CP{19} = [find(strcmp(labelR,'Ym90')) find(strcmp(labelR,'X'))];
CP{20} = [find(strcmp(labelR,'X90')) find(strcmp(labelR,'Y'))];
CP{21} = [find(strcmp(labelR,'X90')) find(strcmp(labelR,'Ym90')) find(strcmp(labelR,'X90'))];
CP{22} = find(strcmp(labelR,'Y90'));
CP{23} = [find(strcmp(labelR,'Xm90')) find(strcmp(labelR,'Y'))];
CP{24} = [find(strcmp(labelR,'X90')) find(strcmp(labelR,'Y90')) find(strcmp(labelR,'Xm90'))];

rhoIn=zeros(2); rhoIn(1,1)=1;
finalSeq = cell(length(nbrClif),1);
listNbrGatesInCliffords = cell(length(nbrClif),1);
for indexSeq = 1:length(nbrClif)
    % Chose random sequence of Cliffords
    elements = randi(24,[1,nbrClif(indexSeq)]);
    
    if isempty(elements)
        sequence = 1;
    else
        % calculate the final gate matrix from the composition of cliffords
        T = eye(2);
        for indexClifford = 1:length(elements)
            T = C{elements(indexClifford)}*T;
        end
        
        % search for the final Clifford
        tempC = T';
        finalC = 0;
        %     for ii = 1:24
        %         if norm(tempC-C{ii}) < 1e-5
        %             finalC = ii;
        %             break
        %         end
        %     end
        for ii = 1:24
            rho = C{ii}*T*rhoIn*T'*C{ii}';
            if norm(rho-rhoIn) < 1e-5
                finalC = ii;
                break
            end
        end
        if finalC == 0
            fprint('here')
        end
        
        sequence = [elements finalC];
        
    end
    
    % build the sequence in terms of Puali rotations
    for ii = 1: length(sequence)
        finalSeq{indexSeq} = [finalSeq{indexSeq} CP{sequence(ii)}];
				listNbrGatesInCliffords{indexSeq} = [listNbrGatesInCliffords{indexSeq} length(CP{sequence(ii)})];
    end
end


if saveFile == 1
    % save data on file
    saveascii('nbrCliffords={',file);
    saveascii(nbrClif,file,0,'a');
    saveascii('}',file,'a');
    
    saveascii('RBlist={',file, 'a');
    for indexSeq = 1:length(finalSeq)-1
        saveascii(['{' saveascii(finalSeq{indexSeq}, 0, ',') '},'], file,'a')
    end
    saveascii(['{' saveascii(finalSeq{end},0, ',') '}'], file,'a')
    saveascii('}',file,'a');
		
		saveascii('NbrGatesInClif={',file, 'a');
    for indexSeq = 1:length(listNbrGatesInCliffords)-1
        saveascii(['{' saveascii(listNbrGatesInCliffords{indexSeq}, 0, ',') '},'], file,'a')
    end
    saveascii(['{' saveascii(listNbrGatesInCliffords{end},0, ',') '}'], file,'a')
    saveascii('}',file,'a');
end
