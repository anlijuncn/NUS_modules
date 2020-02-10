dot1 = [1 0 0];
dot2 = [1 1 0];
dot3 = [1 0 1];
dot4 = [0 1 1];
line = [-1 1.5];
outpath = 'Q3_NAND_offline.png';
fig_title = 'Logic function for NAND with selection of weights by off-line calculaion';
Plot_dots_line(dot1, dot2, dot3, dot4, line, outpath, fig_title);
function Plot_dots_line(dot1, dot2, dot3, dot4, line, outpath, fig_title)
    % Plot_dots_line(dot1, dot2, dot3, dot4, line, outpath)

    % The function is to plot dots with different classes 
    % Inputs:
    %     - dot 1: [y x_1 x_2]
    %     - dot 2: [y x_1 x_2]
    %     - dot 3: [y x_1 x_2]
    %     - dot 4: [y x_1 x_2]
    %     for dots, y is class label, x_1 and x_2 are coordinates 
    %     - line: 
    %            line could separate dots
    %            [k b], x_2 = k*x_1 + b
    %     - outpath: Outpath for saving figure
    %     - fig_title: Title of figure
    % Output:
    %     A figure saved in outpath
    % Written by AN Lijun for EE5904
    
    % we firstly define arrays saving 2 classes dots
    class_0_x1 = [];
    class_0_x2 = [];
    class_1_x1 = [];
    class_1_x2 = [];
    % check which dot is beloging to which class
    % dot1
    if dot1(1) == 0
        class_0_x1 = [class_0_x1, dot1(2)];
        class_0_x2 = [class_0_x2, dot1(3)];
    else
        class_1_x1 = [class_1_x1, dot1(2)];
        class_1_x2 = [class_1_x2, dot1(3)];
    end
    % dot2
    if dot2(1) == 0
        class_0_x1 = [class_0_x1, dot2(2)];
        class_0_x2 = [class_0_x2, dot2(3)];
    else
        class_1_x1 = [class_1_x1, dot2(2)];
        class_1_x2 = [class_1_x2, dot2(3)];
    end
    % dot3
    if dot3(1) == 0
        class_0_x1 = [class_0_x1, dot3(2)];
        class_0_x2 = [class_0_x2, dot3(3)];
    else
        class_1_x1 = [class_1_x1, dot3(2)];
        class_1_x2 = [class_1_x2, dot3(3)];
    end
    % dot 4
    if dot4(1) == 0
        class_0_x1 = [class_0_x1, dot4(2)];
        class_0_x2 = [class_0_x2, dot4(3)];
    else
        class_1_x1 = [class_1_x1, dot4(2)];
        class_1_x2 = [class_1_x2, dot4(3)];
    end
    % draw scatter and line in one figure
    sz= 250;
    scatter(class_0_x1, class_0_x2, sz, 'o');
    hold on 
    scatter(class_1_x1, class_1_x2, sz, 'd');
    hold on 
    hline = refline(line(1), line(2));
    hline.Color = 'k';
    % add legend
    legend('Class 0', 'Class 1', 'Separator: x_2 = -x_1 + 1.5')
    % add axis label 
    xlabel('x_1')
    ylabel('x_2')
    % add figure title
    title(fig_title)
    % save figure
    saveas(gcf, outpath);
end




