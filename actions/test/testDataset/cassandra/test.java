    public void paintComponent(Graphics g){
        clear(g);
        Graphics2D g2d = (Graphics2D)g;
        g2d.setRenderingHints(defaultHints);
        g2d.setPaint(ringColor);
        g2d.setStroke(new BasicStroke(1));
        int dia = getWidth() > getHeight() ? getHeight() - padding : getWidth() - padding;
        int radius = dia / 2;
        double ringX = (getWidth() / 2) - radius;
        double ringY = (getHeight() / 2) - radius;
        Ellipse2D.Double ring = new Ellipse2D.Double(ringX, ringY, dia, dia);
        g2d.draw(ring);
        List<Node> nodes = ringModel.getNodes();
        double current = 0;
        double increment = (2 * Math.PI) / nodes.size();
        for (Node node : nodes){
            Image nodeImage = null;
            switch (node.nodeStatus){
                case ISSEED:
                    nodeImage = nodeImageSeed;
                    break;
                case OK:
                    nodeImage = nodeImageOk;
                    break;
                case SHORT: case LONG:
                    nodeImage = nodeImageShort;
                    break;
                case UNKNOWN:
                    nodeImage = nodeImageUnknown;
                    break;
            }
            double x = Math.cos(current) * radius + (ring.getCenterX() - (nodeDiameter / 2));
            double y = Math.sin(current) * radius + (ring.getCenterY() - (nodeDiameter / 2));
            current = current + increment;
            g2d.drawImage(nodeImage, (int)x, (int)y, null);
            if (node.isSelected()){
                g2d.setPaint(ringColor);
                RoundRectangle2D.Double outline = new RoundRectangle2D.Double(
                        (x - 2), (y - 2), (nodeDiameter + 4), (nodeDiameter + 4), 10, 10);
                g2d.draw(outline);
            }
            g2d.setPaint(fontColor);
            g2d.drawString(node.getHost(), (int)x - (nodeDiameter / 2), (int)y - 5);
        }
        if (isVerifying){
            int msgWidth = g2d.getFontMetrics().charsWidth(verificationMessage, 0, verificationMessage.length);
            g2d.setColor(new Color(128, 128, 128, 128));
            g2d.fillRect(0, 0, getWidth(), getHeight());
            g2d.setColor(fontColor);
            g2d.drawChars(verificationMessage, 0, verificationMessage.length, getWidth()/2 - msgWidth/2, getHeight()/2);
        }
    }