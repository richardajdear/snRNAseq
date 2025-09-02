library(ggplot2)
library(ggh4x)

plot_cells <- function(to_plot, facet_colors=NULL, nrow=1) {
    p <- to_plot %>% 
    ggplot(aes(x=Age_log2, y=zscore, color=zscore)) +
    # geom_jitter(size=.1, width=.1, alpha=.3) +
    geom_point(size=.1, alpha=.2) +
    geom_smooth() +
    scale_x_continuous(
            name = 'Age',
            breaks = log2(1+c(0,1,2,5,9,25,60)),
            labels = function(x) round(2^x-1, 1)
        ) +
    # scale_color_paletteer_c("grDevices::Plasma", 
    #         breaks = c(0,20,40,60,80,100)) +
    scale_color_paletteer_c("grDevices::Viridis") +
    ggtitle('Cell z-scores')

    if (!is.null(facet_colors)) {
        strip = strip_themed(background_x = elem_list_rect(fill=facet_colors))
        p <- p + facet_wrap2(~TF, nrow=nrow, scales='free_y', strip=strip)
    } else {
        p <- p + facet_wrap(~TF, nrow=nrow, scales='free_y')
    }

    return(p)
}

plot_boxes <- function(to_plot, facet_colors=NULL, nrow=1) {
    p <- to_plot %>% 
    ggplot(aes(x=Age_bin, y=zscore)) +
    geom_boxplot(aes(fill=after_stat(middle)), alpha=.3) +
    stat_summary(
        fun = median,
        geom = 'line',
        aes(group = 1), 
        color='blue', size=1,
        position = position_dodge(width = 0.85) 
    ) +
    scale_fill_paletteer_c("grDevices::Viridis", oob=squish, name='zscore') +
    ggtitle('Age bins of cell z-scores')

    if (!is.null(facet_colors)) {
        strip = strip_themed(background_x = elem_list_rect(fill=facet_colors))
        p <- p + facet_wrap2(~TF, nrow=nrow, scales='free_y', strip=strip)
    } else {
        p <- p + facet_wrap(~TF, nrow=nrow, scales='free_y')
    }

    return(p)
}

plot_boxes_donor <- function(to_plot, facet_colors=NULL, nrow=1) {
    p <- to_plot %>% 
    group_by(TF, Individual, Age_bin) %>% 
    summarise(zscore = mean(zscore, na.rm=TRUE)) %>%
    ggplot(aes(x=Age_bin, y=zscore)) + 
    geom_boxplot(aes(fill=after_stat(middle)), alpha=.3) +
    stat_summary(
        fun = median,
        geom = 'line',
        aes(group = 1), 
        color='blue', size=1,
        position = position_dodge(width = 0.85) 
    ) +
    scale_fill_paletteer_c("grDevices::Viridis", oob=squish, name='zscore') +
    ggtitle('Age bins of donor-averaged z-scores')

    if (!is.null(facet_colors)) {
        strip = strip_themed(background_x = elem_list_rect(fill=facet_colors))
        p <- p + facet_wrap2(~TF, nrow=nrow, scales='free_y', strip=strip)
    } else {
        p <- p + facet_wrap(~TF, nrow=nrow, scales='free_y')
    }

    return(p)
}

plot_boxes_random <- function(to_plot, facet_colors=NULL, nrow=1) {
    p <- to_plot %>% 
    mutate(Random_bin = sample(1:20, n(), replace=T)) %>% 
    group_by(TF, Random_bin, Age_bin) %>% 
    summarise(zscore = mean(zscore, na.rm=TRUE)) %>%
    ggplot(aes(x=Age_bin, y=zscore)) + 
    geom_boxplot(aes(fill=after_stat(middle)), alpha=.3) +
    stat_summary(
        fun = median,
        geom = 'line',
        aes(group = 1), 
        color='blue', size=1,
        position = position_dodge(width = 0.85) 
    ) +
    scale_fill_paletteer_c("grDevices::Viridis", oob=squish, name='zscore') +
    ggtitle('Age bins of random-100-averaged z-scores')

    if (!is.null(facet_colors)) {
        strip = strip_themed(background_x = elem_list_rect(fill=facet_colors))
        p <- p + facet_wrap2(~TF, nrow=nrow, scales='free_y', strip=strip)
    } else {
        p <- p + facet_wrap(~TF, nrow=nrow, scales='free_y')
    }

    return(p)
}
