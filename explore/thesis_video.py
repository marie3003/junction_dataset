from manim import *


class ThesisVideo(Scene):
    def construct(self):
        self.intro()
        self.antibiotic_resistance_demo()
        self.horizontal_gene_transfer()

    def intro(self):
        title = Text("Accessory Genome Evolution in E. coli ST131", font_size=46)
        subtitle = Text("and its relevance for Sustainable Development", font_size=30)
        subtitle.next_to(title, DOWN)

        self.play(FadeIn(title, shift=UP), FadeIn(subtitle, shift=UP))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))

    def antibiotic_resistance_demo(self):
        title = Text(
            "Antimicrobial resistance is a major global health challenge",
            font_size=32,
        ).to_edge(UP)
        self.add(title)

        n_bacteria = 5
        spacing = 1.9   # enough room so flagella don't overlap
        y_top = 1.2
        y_bot = -1.3
        x_offset_res = 0.6  # bottom row shifted right

        x_start_top = -5.5
        x_start_bot = -4.5
        top_positions = [RIGHT * (x_start_top + i * spacing) + UP * y_top for i in range(n_bacteria)]
        bot_positions = [RIGHT * (x_start_bot + x_offset_res + i * spacing) + UP * y_bot
                         for i in range(n_bacteria)]

        def make_bacterium(pos, color=WHITE):
            body = Ellipse(width=0.9, height=0.6, color=color).move_to(pos)
            nucleus = Ellipse(width=0.35, height=0.22, color=color, fill_opacity=0.4).move_to(pos)
            # Flagellum: wavy tail going down-right so it doesn't reach the next bacterium
            p0 = pos + RIGHT * 0.45
            flagellum = CubicBezier(
                p0,
                p0 + RIGHT * 0.3 + DOWN * 0.25,
                p0 + RIGHT * 0.6 + UP * 0.15,
                p0 + RIGHT * 0.85 + DOWN * 0.1,
                color=color,
            )
            return VGroup(body, nucleus, flagellum)

        def make_drug(pos, color="#a6444f"):
            pill = RoundedRectangle(width=0.45, height=0.22, corner_radius=0.1,
                                    color=color, fill_color=color, fill_opacity=0.9)
            pill.move_to(RIGHT * 7 + UP * pos[1])
            return pill

        def make_cross(center):
            # Cross centered on the bacterium body center, not the full VGroup
            r = 0.5
            return VGroup(
                Line(center + UL * r, center + DR * r, color="#a6444f", stroke_width=4),
                Line(center + UR * r, center + DL * r, color="#a6444f", stroke_width=4),
            )

        # --- Phase 1: top row — bacteria die ---
        bacteria = VGroup(*[make_bacterium(p) for p in top_positions])
        self.play(LaggedStart(*[GrowFromCenter(b) for b in bacteria], lag_ratio=0.15))
        self.wait(0.3)

        # bacteria[i][0] is the body ellipse — target its center, not the VGroup center
        body_centers_top = [bacteria[i][0].get_center() for i in range(n_bacteria)]
        drugs1 = VGroup(*[make_drug(p) for p in top_positions])
        self.add(drugs1)
        self.play(LaggedStart(*[
            d.animate.move_to(body_centers_top[i])
            for i, d in enumerate(drugs1)
        ], lag_ratio=0.1), run_time=1.4)
        self.wait(0.2)

        crosses = VGroup(*[make_cross(top_positions[i]) for i in range(n_bacteria)])
        self.play(
            LaggedStart(*[Create(c) for c in crosses], lag_ratio=0.1),
            FadeOut(drugs1),
            run_time=0.8,
        )
        self.wait(0.5)

        # --- Phase 2: bottom row — resistant bacteria, drugs bounce off ---
        resistant_bacteria = VGroup(*[make_bacterium(p, color="#57a8b8") for p in bot_positions])
        badges = VGroup(*[
            Text("R", font_size=13, color="#57a8b8").move_to(p + UP * 0.36)
            for p in bot_positions
        ])
        self.play(LaggedStart(*[GrowFromCenter(b) for b in resistant_bacteria], lag_ratio=0.15))
        self.play(FadeIn(badges))
        self.wait(0.3)

        body_centers_bot = [resistant_bacteria[i][0].get_center() for i in range(n_bacteria)]
        drugs2 = VGroup(*[make_drug(p) for p in bot_positions])
        self.add(drugs2)
        self.play(LaggedStart(*[
            d.animate.move_to(body_centers_bot[i])
            for i, d in enumerate(drugs2)
        ], lag_ratio=0.1), run_time=1.2)
        self.wait(0.2)

        self.play(LaggedStart(*[
            d.animate.move_to(RIGHT * 8 + UP * bot_positions[i][1])
            for i, d in enumerate(drugs2)
        ], lag_ratio=0.08), run_time=0.9)

        self.wait(2)

        self.play(
            FadeOut(title),
            FadeOut(bacteria), FadeOut(crosses),
            FadeOut(resistant_bacteria), FadeOut(badges),
        )

    def horizontal_gene_transfer(self):
        heading = Text(
            "Bacteria evolve via Horizontal Gene Transfer",
            font_size=32,
        ).to_edge(UP)

        chrom_y = -0.2
        h = 0.42
        gap = 0.08
        grey  = "#7a7a7a"
        teal  = "#57a8b8"
        red   = "#a6444f"
        blue  = "#397398"
        purp  = "#80557e"
        pink  = "#d991b4"

        def make_block(w, color):
            b = RoundedRectangle(width=w, height=h, corner_radius=0.06,
                                 color=color, fill_color=color, fill_opacity=1)
            return b

        def layout(specs, y=0):
            total_w = sum(w for w, _ in specs) + gap * (len(specs) - 1)
            x = -total_w / 2
            blocks = []
            for w, color in specs:
                b = make_block(w, color)
                b.move_to(RIGHT * (x + w / 2) + UP * y)
                x += w + gap
                blocks.append(b)
            return blocks

        old_specs = [
            (0.4, grey),  # 0 left cap
            (0.9, grey),  # 1 left gene
            (0.6, teal),  # 2 anchor_L
            (0.55, red),  # 3 mid 1
            (0.75, blue), # 4 mid 2
            (0.55, red),  # 5 mid 3
            (0.6, teal),  # 6 anchor_R
            (0.9, grey),  # 7 right gene
            (0.4, grey),  # 8 right cap
        ]
        blks = layout(old_specs, y=chrom_y)
        chromosome = VGroup(*blks)

        # Bacterium oval around the chromosome
        cell = Ellipse(width=8.0, height=2.6,
                       color="#397398", fill_color="#b5d2f2", fill_opacity=0.15)
        cell.move_to(UP * chrom_y)

        # Everything appears instantly when slide starts
        self.add(heading, cell, chromosome)
        self.wait(0.5)

        left_group  = VGroup(blks[0], blks[1], blks[2])
        old_middle  = VGroup(blks[3], blks[4], blks[5])
        right_group = VGroup(blks[6], blks[7], blks[8])

        # Old middle flies up and out
        self.play(old_middle.animate.shift(UP * 3.5), run_time=0.8)
        self.remove(old_middle)

        # New shorter middle arrives from below
        new_specs = [(0.55, purp), (0.55, pink)]
        new_mid_total = sum(w for w, _ in new_specs) + gap * (len(new_specs) - 1)
        new_blks = layout(new_specs, y=chrom_y - 3.5)
        new_middle = VGroup(*new_blks)
        self.add(new_middle)

        # Compute how far the ends need to slide inward
        anchor_l_target_x = -new_mid_total / 2 - gap - blks[2].width / 2
        anchor_r_target_x =  new_mid_total / 2 + gap + blks[6].width / 2
        dx_l = anchor_l_target_x - blks[2].get_center()[0]
        dx_r = anchor_r_target_x - blks[6].get_center()[0]

        # New middle slides up while ends close in simultaneously
        self.play(
            new_middle.animate.shift(UP * 3.5),
            left_group.animate.shift(RIGHT * dx_l),
            right_group.animate.shift(RIGHT * dx_r),
            run_time=1.1,
        )

        self.wait(2)
        self.play(
            FadeOut(heading),
            FadeOut(cell),
            FadeOut(left_group), FadeOut(right_group), FadeOut(new_middle),
        )
