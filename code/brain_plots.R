library(ggseg)
library(ggsegGlasser)

plot_maps <- function(maps, atlas='dk', title="", ncol=3, facet='w', 
                      spacing_x=0.5, spacing_y=0.5,
                    #   position="stacked",
                      colors=rev(brewer.rdbu(100)), 
                      colorscale='fixed', limits=c(-3,3),
                      labels=c('-3σ','+3σ'), name='') {
    
    if ("label" %in% colnames(maps)) {
        maps <- maps %>% remove_rownames %>% column_to_rownames('label')
    }

    df <- maps %>%
        rownames_to_column %>%
        mutate(region = rowname) %>%
        mutate(label = recode(rowname,'7Pl'='7PL')) %>%
        select(-rowname) %>%
        gather('map', 'value', -region) %>%
        mutate_at(vars(map), ~ factor(., levels=unique(.))) %>%
        group_by(map)
    
    # set scale limits at 99th percentile
    m_max <- pmax(
        df %>% .$value %>% quantile(.99, na.rm=T) %>% abs,
        df %>% .$value %>% quantile(.01, na.rm=T) %>% abs
    )

    if (colorscale == 'fixed') {
        m_min <- limits[1]
        m_max <- limits[2]
    } else if (colorscale=='symmetric') {
        m_min <- -m_max
    } else if (colorscale=='zero') {
        m_min <- 0
    } else if (colorscale=='absolute') {
        m_min <- df %>% .$value %>% quantile(.01)
    } else {
        print("Invalid colorscale")
    }

    # set manual axis labels if desired
    if (length(labels)>1) {
        labels = labels
    } else if (labels=='none') {
        labels = c(round(m_min,2),round(m_max,2))
    } else if (labels=='centile') {
        labels = c(round(m_min+0.5,2),round(m_max+0.5,2))
    }

    p <- df %>% ggseg(
        atlas=atlas,
        hemi='left',
        # mapping=aes(fill=value, geometry=geometry, hemi=hemi, side=side, type=type),
        mapping=aes(fill=value),
        # position=position_brain(c('left lateral','left medial')),
        # position=position,
        colour='grey', size=.1,
        show.legend=T
        ) + 
    # facet_wrap(~map, ncol=ncol, dir="v") +
    scale_fill_gradientn(
        colors=colors, 
        limits=c(m_min,m_max), oob=squish, breaks=c(m_min,m_max), 
        labels=labels, 
        guide=guide_colorbar(barheight=.3, barwidth=3),
        name=name
    ) +
    theme_void() + 
    theme(legend.position='bottom',
          legend.title=element_text(vjust=1),
          panel.spacing.x=unit(spacing_x,'lines'),
          panel.spacing.y=unit(spacing_y,'lines'),
          strip.text.x=element_text(vjust=1, size=5),
          text=element_text(size=5),
          strip.clip='off',
          plot.title=element_text(hjust=0.5),
          plot.tag.position = c(0,1)
    ) +
    ggtitle(title) + xlab("") + ylab("")
    
    if (facet=='h') {
        p + facet_grid(.~map)
    } else if (facet=='v') {
        p + facet_grid(map~.)
    } else if (facet=='w') {
        # Special facet wrap to remove clipping, requires ggh4x package
        # p + facet_wrap2(~map, ncol=ncol, dir="v",
        #         strip=strip_vanilla(clip='off')
        p + facet_wrap(~map, ncol=ncol, dir="v")
    }
}